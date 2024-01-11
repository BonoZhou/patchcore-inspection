"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import time
LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
        self.anomaly_score_num_nn = anomaly_score_num_nn
        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        # print(input_shape,feature_dimensions)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.features = None
        self.time = {'embed_feature_aggregator':[],
                     'embed_preprocessing':[],
                     'embed_preadapt_aggregator':[],
                     'embed_reshape':[],
                     'embed':[],
                     'predict_process':[],
                     'embed_all':[]} # time calculate


    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""
        allstart = time.time()
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features


        start = time.time()
        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        self.time['embed_feature_aggregator'].append(time.time()-start)

        start = time.time()
        features = [features[layer] for layer in self.layers_to_extract_from] # 'layer2', 'layer3'
        # print(features[0].shape,features[1].shape)
        # [2, 512, 28, 28] [2, 1024, 14, 14]
        # 返回两层特征
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        # torch.Size([2, 784, 512, 3, 3]) torch.Size([2, 196, 1024, 3, 3])
        # print(features[0][0].shape,features[1][0].shape)

        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]
        # print(patch_shapes) [[28, 28], [14, 14]]
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        # print(features[0].shape,features[1].shape) torch.Size([1568, 512, 3, 3]) torch.Size([1568, 1024, 3, 3])
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        self.time['embed_reshape'].append(time.time()-start)

        #time calculate
        start = time.time()
        features = self.forward_modules["preprocessing"](features)
        self.time['embed_preprocessing'].append(time.time()-start)
        # print(features.shape)[1568, 2, 1024]
        start = time.time()
        features = self.forward_modules["preadapt_aggregator"](features)
        self.time['embed_preadapt_aggregator'].append(time.time()-start)
        # print(features.shape)[1568, 2, 1024]
        self.time['embed_all'].append(time.time()-allstart)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                
                batchsize = image.shape[0] # 2
                
                feature_batch_image = np.array(_image_to_features(image))
                feature_batch_image = feature_batch_image.reshape(batchsize,-1,feature_batch_image.shape[-1]) # 2,-1,1024

                features.append(feature_batch_image)

        features = np.concatenate(features, axis=0) # 209,784,1024
        self.features = torch.Tensor(features.transpose(1,0,2)) # 784,209,1024
        # self.anomaly_scorer.fit(detection_features=[features])


    @staticmethod
    def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format.
        Resulting matrix is indexed by x vectors in rows and y vectors in columns.

        Args:
            x: input tensor 1
            y: input tensor 2

        Returns:
            Matrix of distances between row vectors in x and y.
        """
        y = y.reshape(-1,y.shape[-1]).cuda()
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        res = res.clamp_min_(0).sqrt_()
        return res

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int, memory_bank=None):
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        if memory_bank==None:
            distances = self.euclidean_dist(embedding, self.features)
        else:
            distances = self.euclidean_dist(embedding, memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        
        return patch_scores, locations # 别忘了batchsize


    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        patch_memory = self.features.unsqueeze(2).cuda() # -1,209,1,1024
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            inft = []

            for i in self.time:
                self.time[i].clear()
            for image in data_iterator:
                start = time.time()
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image,patch_memory=patch_memory)
                inft.append(time.time()-start)

                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
            #print time
            print('inference time:',np.mean(inft),'fps:',1/np.mean(inft))
            for i in self.time:
                print(i,":",np.sum(self.time[i]))

        return scores, masks, labels_gt, masks_gt

    def _predict(self, images,patch_memory=None):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():

            #cal time
            start = time.time()
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            self.time['embed'].append(time.time()-start)

            features = np.asarray(features) # 1568,1024
            
            
            start = time.time()
            # patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            # Teng add
            #--------------------------------------------------------------------

            
            features = features.reshape(batchsize,-1,features.shape[-1]) # 2,-1,1024
            features = features.transpose(1,0,2) # -1,2,1024
            features = torch.Tensor(features)



            
            features = features.unsqueeze(1).cuda() # -1,1,2,1024
            #patch_memory = self.features.unsqueeze(2).cuda() # -1,209,1,1024


            
            features = features.expand(-1,patch_memory.shape[1],-1,-1) # -1,209,2,1024
            patch_memory = patch_memory.expand(-1,-1,features.shape[2],-1) # -1,209,2,1024





            
            distances = torch.norm(features-patch_memory,dim=3)
            min_distances,_ = torch.min(distances,dim=1)


            image_scores = min_distances.reshape(-1,batchsize).cpu()
            # image_scores = []
            # for i in (range(features.shape[0])):
            #     # image_scores.append(self.anomaly_scorer.predict([features[i]])[0]) # 2
            #     image_scores.append(np.array(self.nearest_neighbors((features[i]), self.anomaly_score_num_nn, self.features[i])[0].cpu()))
            #     # image_scores.append(np.array(self.nearest_neighbors((features[i]), self.anomaly_score_num_nn)[0].cpu()))
            
            image_scores = np.asarray(image_scores) # (28*28,2)
            

            image_scores = image_scores.transpose(1,0).reshape(-1)
            patch_scores = image_scores
            
            #-------------------------------------------------------------------

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            # print(image_scores.shape)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            # print(image_scores.shape)
            image_scores = self.patch_maker.score(image_scores)
            # print(image_scores)
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            # print(scales)
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            # print(masks[0].shape)
            self.time['predict_process'].append(time.time()-start)
        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
