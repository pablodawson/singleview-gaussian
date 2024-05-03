import torch
from gmflow.gmflow.gmflow import GMFlow
import numpy as np
import os
from gmflow.utils.utils import InputPadder
import torch.nn.functional as F
from PIL import Image

class FlowEstimator:
    def __init__(self, model_path, device, baseline=0.068, fov=80):
        self.model = self.load_model(model_path, device)
        self.device = device
        self.baseline = baseline
        self.fov = fov
    
    def load_model(self, model_path, device):
        loc = 'cuda:{}'.format(0)
        checkpoint = torch.load(model_path, map_location=device)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model = GMFlow().to(device)
        model_without_ddp = model
        model_without_ddp.load_state_dict(weights)
        model_without_ddp.eval()

        return model_without_ddp

    def process_image(self, img):

        if type(img) == np.ndarray:
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)

        inference_size = [384, 512]
        #inference_size = None

        if inference_size is None:
            padder = InputPadder(img.shape, padding_factor=16)
            img = padder.pad(img[None].cuda())
        else:
            img = img[None].cuda()

        # resize before inference
        if inference_size is not None:
            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            #ori_size = img.shape[-2:]
            img = F.interpolate(img, size=inference_size, mode='bilinear',
                                   align_corners=True)
        
        return img[0]
            
    def estimate_flow(self, img0, img1):
        
        og_size = img0.shape[:2]
        
        img0 = self.process_image(img0)
        img1 = self.process_image(img1)

        new_size = img0.shape[-2:]
        
        #imgs -> tensors with range 0-255
        results_dict = self.model(img0, img1, [2], [-1], [-1])
        flow_preds = results_dict['flow_preds']

        # Resize after inference
        flow_pred = F.interpolate(flow_preds[0], size=og_size, mode='bilinear',
                                   align_corners=True)
        
        # Compensate for changed resolution
        flow_pred = flow_pred * torch.tensor([og_size[1]/new_size[1], og_size[0]/new_size[0]]).view(1, 2, 1, 1).to(flow_pred.device)
        flow_np = flow_pred.cpu().detach().numpy()
        flow_np = flow_np[0].transpose(1, 2, 0)

        return flow_np
    
    def get_depth(self, imgL, imgR):
        flowLR = -self.estimate_flow(imgL, imgR)
        flowRL = self.estimate_flow(imgR, imgL)
        disparityL = flowLR[:, :, 0]
        disparityR = flowRL[:, :, 0]
        
        inference_w = imgL.shape[0]
        fx = inference_w/(2*np.tan(np.radians(self.fov)/2))
        
        disparityL = np.clip(disparityL, 10, 100)
        disparityR = np.clip(disparityR, 10, 100)
        
        depthL = fx*(self.baseline / disparityL)
        depthR = fx*(self.baseline / disparityR)
        
        return depthL, depthR

def vis(flow):
    disparity = flow[:, :, 0]
    disparity = (disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity))
    disparity = (disparity * 255).astype(np.uint8)
    return disparity



if __name__ == '__main__':
    model_path = "gmflow/pretrained/gmflow_things-e9887eda.pth"
    device = 'cuda:0'
    flow_estimator = FlowEstimator(model_path, device)

    img0 = np.array(Image.open("output/images/cam00.png"))
    img1 = np.array(Image.open("output/images/cam01.png"))

    flowLR = -flow_estimator.estimate_flow(img0, img1)
    flowRL = flow_estimator.estimate_flow(img1, img0)
    
    Image.fromarray(vis(flowLR)).save("disparityL.png")
    Image.fromarray(vis(flowRL)).save("disparityR.png")
