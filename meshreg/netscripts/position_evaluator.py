import numpy as np
import torch
from meshreg.models.utils import torch2numpy
from scipy.spatial.distance import pdist
from scipy import linalg

#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.


class ZimEval:
    """ Util class for evaluation networks.
    """

    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_pred, keypoint_vis=None):
        """
        Used to feed data to the class.
        Stores the euclidean distance between gt and pred, when it is visible.
        """
        
        keypoint_gt = np.squeeze(torch2numpy(keypoint_gt))
        keypoint_pred = np.squeeze(torch2numpy(keypoint_pred))

        if keypoint_vis is None:
            keypoint_vis = np.ones_like(keypoint_gt[:, 0])
        keypoint_vis = np.squeeze(keypoint_vis).astype("bool")

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])
        return np.mean(euclidean_dist)

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype("float"))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)
        # Display error per keypoint
        epe_mean_joint = epe_mean_all
        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(
            np.array(pck_curve_all), 0
        )  # mean only over keypoints

        return (
            epe_mean_all,
            epe_mean_joint,
            epe_median_all,
            auc_all,
            pck_curve_all,
            thresholds,
        )


def feed_evaluators_hand(evaluators, pred, gt, weights, tag, center_idx): 
    pred_joints3d = torch2numpy(pred)
    gt_joints3d=torch2numpy(gt)
    weights=torch2numpy(weights)

    if center_idx is not None:
        assert center_idx==0
        gt_joints3d_cent = gt_joints3d - gt_joints3d[:, center_idx : center_idx + 1]
        pred_joints3d_cent = pred_joints3d - pred_joints3d[:, center_idx : center_idx + 1]
        
        for cid in range(0,pred_joints3d.shape[0]):
            gt_joints,pred_joints=gt_joints3d_cent[cid],pred_joints3d_cent[cid]
            if weights[cid]>1e-4:
                evaluators[f"{tag}joints3d_cent"].feed(gt_joints[1:,], pred_joints[1:,])
    

    for cid in range(0,pred_joints3d.shape[0]):
        gt_joints,pred_joints=gt_joints3d[cid],pred_joints3d[cid]
        if weights[cid]>1e-4:
            evaluators[f"{tag}joints3d"].feed(gt_joints, pred_joints)






def parse_evaluators(evaluators, config=None):
    """
    Parse evaluators for which PCK curves and other statistics
    must be computed
    """
    if config is None:
        config = {"joints3d": [0, 0.08, 33],
            "left_joints3d": [0, 0.09, 37],
            "right_joints3d": [0, 0.09, 37],
            
            "left_joints3d_cent": [0, 0.05, 21],
            "right_joints3d_cent": [0, 0.05, 21],
            "joints3d_cent": [0, 0.05, 21],}
    eval_results = {}
    for evaluator_name, evaluator in evaluators.items():
        start, end, steps = [config[evaluator_name][idx] for idx in range(3)]
        (epe_mean, epe_mean_joints, epe_median, auc, pck_curve, thresholds) = evaluator.get_measures(start, end, steps)
        eval_results[evaluator_name] = {
            "epe_mean": epe_mean,
            "epe_mean_joints": epe_mean_joints,
            "epe_median": epe_median,
            "auc": auc,
            "thresholds": thresholds,
            "pck_curve": pck_curve,
        }
        
    return eval_results


################MyMEPE######################

class MyMEPE:
    def __init__(self):
        self.sum_hand_error=None
        self.num_samples=None
    
    def feed(self,batch_seq_pred3d,batch_seq_gt3d, batch_seq_weights):
        if self.sum_hand_error is None:
            self.sum_hand_error=np.zeros((batch_seq_gt3d.shape[1]))
            self.num_samples=np.zeros((batch_seq_gt3d.shape[1]))
        batch_seq_pred3d=torch2numpy(batch_seq_pred3d)
        batch_seq_gt3d=torch2numpy(batch_seq_gt3d)
        batch_seq_weights=torch2numpy(batch_seq_weights)
        batch_seq_dist=np.linalg.norm(batch_seq_pred3d-batch_seq_gt3d,axis=-1)#dim-wise, [bs,len_seq,num_joints]

        for idx in range(0,batch_seq_pred3d.shape[0]):
            cdist=batch_seq_dist[idx].mean(axis=-1)#joint-wise,[len_seq]
            cweight=np.ones_like(cdist)  if batch_seq_weights is None else batch_seq_weights[idx]
            self.sum_hand_error+=np.multiply(cdist,cweight)
            self.num_samples+=cweight

    def aggregate(self):
        self.num_samples=np.where(self.num_samples<1,1,self.num_samples)#max(1,self.num_samples)
        print(self.num_samples)

        epe_mean=np.zeros_like(self.sum_hand_error)
        for i in range(0,self.sum_hand_error.shape[0]):
            epe_mean[i]=self.sum_hand_error[i]/self.num_samples[i]
        
        return  {"epe_mean": epe_mean.mean(),"per_frame_epe_mean":epe_mean}

        
def feed_mymepe_evaluators_hands(evaluator,batch_seq_joints3d_out, batch_seq_joints3d_gt,batch_seq_weights,valid_joints):
    for hid in range(0,2):
        batch_seq_cgt=batch_seq_joints3d_gt[:,:,hid*21:(hid+1)*21]
        batch_seq_cout=batch_seq_joints3d_out[:,:,hid*21:(hid+1)*21]
            
        tag="left_" if hid==0 else "right_"        
        if valid_joints is not None:
            batch_seq_cgt=torch.cat([batch_seq_cgt[:,:,idx:idx+1] for idx in valid_joints],2)
            batch_seq_cout=torch.cat([batch_seq_cout[:,:,idx:idx+1] for idx in valid_joints],2)

        evaluator[f"{tag}joints3d"].feed(batch_seq_cgt,batch_seq_cout,batch_seq_weights)
 

def aggregate_and_save(path_to_save,save_dict):
    pred_left,pred_right,pred_left_cent,pred_right_cent=[],[],[],[]
    pred_left_aligned,pred_right_aligned=[],[]
    for pid in range(0,save_dict["ntokens_pred"]):
        pred_left.append(save_dict[f"pred{pid}_left_joints3d_epe_mean"])
        pred_right.append(save_dict[f"pred{pid}_right_joints3d_epe_mean"])
        pred_left_cent.append(save_dict[f"pred{pid}_left_joints3d_local_epe_mean"])
        pred_right_cent.append(save_dict[f"pred{pid}_right_joints3d_local_epe_mean"])
        if f"pred{pid}_right_joints3d_ra_epe_mean" in save_dict:
            pred_left_aligned.append(save_dict[f"pred{pid}_left_joints3d_ra_epe_mean"])
            pred_right_aligned.append(save_dict[f"pred{pid}_right_joints3d_ra_epe_mean"])
    np.savez(path_to_save,
            pred_left=pred_left,pred_right=pred_right,
            pred_left_local=pred_left_cent,pred_right_local=pred_right_cent,
            pred_left_ra=pred_left_aligned,pred_right_ra=pred_right_aligned)


class MyVAE:
    def __init__(self,mode="mean"):
        self.sum_average_pose_error=0
        self.sum_furthest_pose_error=0
        self.sum_average_pose_diversity=0
        self.cnt=0
        self.mode=mode 
        
    def compute_pose_error(self,rs_pred, gt, frame_idx, valid_len):
        #print(rs_pred.shape,gt.shape)#[num_rs,len_seq,joints,3],[1,len_seq,joints,3]
        diff = rs_pred - gt
        dist = np.linalg.norm(diff, axis=-1)#coord-dim,[num_rs,len_seq,joints]
        dist=dist[:,:valid_len]
        #dist2=(np.sum(diff**2,axis=-1)**0.5)
        
        dist=dist.mean(axis=2)# joints-dim
        dist=dist.mean(axis=1) if frame_idx is None else dist[:,frame_idx]#frame-dim frame_idx is None then return average,shape [num_rs]
        dist_to_return=dist.min() if self.mode=="best" else (dist.max() if self.mode=="worst" else dist.mean()) #sample-dim
        return dist_to_return


    def compute_diversity(self,rs_pred,valid_len):#rs_pred[num_rs,len_seq,joints,3]
        rs_pred=rs_pred[:,:valid_len]
        num_samples,len_seq = rs_pred.shape[0], rs_pred.shape[1]
        
        sum_dist=0.
        for i in range(1,num_samples):
            for j in range(0,i):
                cdist=np.linalg.norm(rs_pred[i]-rs_pred[j],axis=-1)#coord-dim, [len_seq,joints]
                sum_dist+=cdist.mean()

        return sum_dist/(num_samples*(num_samples-1)//2)
        
        

    def feed(self,batch_seq_gt,batch_rs_seq_out,batch_seq_weights):
        batch_seq_gt=torch2numpy(batch_seq_gt)
        batch_rs_seq_out=torch2numpy(batch_rs_seq_out)
        batch_seq_weights=torch2numpy(batch_seq_weights)
        num_gts=batch_seq_gt.shape[0]
        
        for gtid in range(0,num_gts):
            if batch_seq_weights is not None and batch_seq_weights[gtid,-1]<1e-6:
                valid_len=np.sum(batch_seq_weights[gtid])
                assert batch_seq_weights[gtid,valid_len-1]>0 and batch_seq_weights[gtid,valid_len]<1e-6
            else:
                valid_len=batch_seq_weights.shape[1]
            
            self.sum_average_pose_error+=self.compute_pose_error(batch_rs_seq_out[gtid],batch_seq_gt[gtid,np.newaxis],frame_idx=None,valid_len=valid_len)
            self.sum_furthest_pose_error+=self.compute_pose_error(batch_rs_seq_out[gtid],batch_seq_gt[gtid,np.newaxis],frame_idx=valid_len-1,valid_len=valid_len)
            self.sum_average_pose_diversity+=self.compute_diversity(batch_rs_seq_out[gtid],valid_len=valid_len)
               
            self.cnt+=1

    def aggregate(self):
        self.cnt=max(self.cnt,1)
        return {"ape":self.sum_average_pose_error/self.cnt,
            "apd":self.sum_average_pose_diversity/self.cnt,
            "fpe":self.sum_furthest_pose_error/self.cnt,}

def feed_myvae_evaluator_hands(evaluator,batch_rs_seq_joints3d_out, batch_seq_joints3d_gt,batch_seq_weights,valid_joints=None):
    for hid in range(0,2):
        batch_rs_seq_out=batch_rs_seq_joints3d_out[:,:,:,hid*21:(hid+1)*21]
        batch_seq_gt=batch_seq_joints3d_gt[:,:,hid*21:(hid+1)*21]

        tag="left_" if hid==0 else "right_"
            
        if valid_joints is not None:
            batch_rs_seq_out=torch.cat([batch_rs_seq_out[:,:,:,idx:idx+1] for idx in valid_joints],3)
            batch_seq_gt=torch.cat([batch_seq_gt[:,:,idx:idx+1] for idx in valid_joints],2)


        evaluator[f"{tag}joints3d"].feed(batch_seq_gt,batch_rs_seq_out,batch_seq_weights)



def parse_evaluators(evaluators):    
    eval_results = {}
    for evaluator_name, evaluator in evaluators.items():
        eval_results[evaluator_name] = evaluator.aggregate()
    return eval_results


class H2OPoseSummary:
    def __init__(self):
        self.result_json={"modality":"RGB"}
    def feed(self,flatten_pred3d,flatten_sample_info,flatten_weights):
        pred3d=torch2numpy(flatten_pred3d)
        weights=torch2numpy(flatten_weights)

        for idx in range(0,pred3d.shape[0]):
            if weights[idx]<1e-4:
                continue
            cseq_idx=flatten_sample_info["seq_idx"][idx]
            seq_key="{:d}".format(int(cseq_idx)+1)
            if not (seq_key in self.result_json.keys()):
                self.result_json[seq_key]={}
            frame_key="{:06d}.txt".format(flatten_sample_info["frame_idx"][idx])

            assert not (frame_key in self.result_json[seq_key].keys()), 'duplicate keys!'
            self.result_json[seq_key][frame_key]=pred3d[idx].flatten().tolist()
            assert len(self.result_json[seq_key][frame_key])==126, 'output hpose len should be 126!'

    
    def save_json(self):
        with open("./hand_poses.json","w") as f:
            json.dump(self.result_json,f)


def calculate_activation_statistics(activations):
    """ Compute mean and covariance of activations"""
    activations = torch2numpy(activations)#activations.cpu().numpy()
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Frechet distance between two multivariate Gaussians
    X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2):
    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    print("Computing FID ...", end='')
    mu1, mu2, sigma1, sigma2 = [np.atleast_1d(x) for x in [mu1, mu2, sigma1, sigma2]]
    assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape, 'Incoherent vector shapes'

    diff = mu1 - mu2
    print("diff",diff.shape)
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    print("covmean",covmean.shape)#[512,512]

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    print('OK!')
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


