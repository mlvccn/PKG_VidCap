import os
import numpy as np
import basic_utils as utils
import json
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from torch import nn
import clip
import pickle
from transformers import AutoTokenizer, AutoModel
import glob
import argparse

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_np(filename):
    with open(filename, "rb") as f:
        return np.load(f)

def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def vatex_sents(params):
    coco_ids = utils.read_lines(params['coco_ids'])
    coco_ids = set(coco_ids)

    full_sents = []
    raw_anno = utils.load_json(os.path.join(params['anno_dir'], 'dataset.json'))['images']
    for data in raw_anno:
        image_id = str(data['imgid'])
        if image_id not in coco_ids:
            continue
        sents = data['sentences']
        for sent in sents:
            sent = sent['raw'].lower().strip(' ').strip('.')
            sent = sent.replace('\n', ' ')
            sent = sent.replace(';', '')
            words = sent.split(' ')[0:20]
            sent = ' '.join(words)
            full_sents.append(sent)

    print(len(full_sents))

    utils.save_lines(full_sents, '/video-text_retrieval/data/vatex_sents_refine.txt')
    return full_sents

def txtfeats(params, full_sents):
    
    tokenizer = AutoTokenizer.from_pretrained("/home/wangtao/video_caption/Transformer/X-Clip/model")
    model = AutoModel.from_pretrained("/home/wangtao/video_caption/Transformer/X-Clip/model")

    print('encode text')
    
    sents_list = ['a photo of ' + line.strip() for line in full_sents]

    feat_sents = np.zeros((len(sents_list), 512)).astype('float32') #(48748, 512)
    batch_size = 64
    num_batches = len(sents_list) // batch_size + 1 #762

    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            if i == num_batches - 1:
                sents = sents_list[i*batch_size:]
            else:
                sents = sents_list[i*batch_size:i*batch_size+batch_size]
            
            if sents == []:
                break

            inputs = tokenizer(sents, padding=True, return_tensors="pt")
            
            text_features = model.get_text_features(**inputs).cpu().numpy()

            feat_sents[i*batch_size:i*batch_size + len(text_features)] = text_features

    save_pickle(feat_sents, '/video-text_retrieval/data/x-clip_vatex_sents.pkl')
    return feat_sents

def simi(params):
    top_k = params['sents_top_k']

    text_features = load_pickle('/video-text_retrieval/data/x-clip_vatex_sents.pkl')
    # text_features = feat_sents
    text_features = torch.as_tensor(text_features).cuda() #(46416, 512)

    raw_anno = utils.load_json(os.path.join(params['anno_dir'], 'dataset.json'))['images']
    coco_ids = utils.read_lines(params['coco_ids'])
    sents_dir = {}
    for data in tqdm(raw_anno):
        vid_id = str(data['imgid'])
        sents_dir[vid_id] = np.array(data['sentids'])

    retrieval_res = {}
    all_vids_path = glob.glob('/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video-text_retrieval/data/vatex/x-clip/*.npy')
    for vid_path in tqdm(all_vids_path):
        _, tempfilename = os.path.split(vid_path)
        vid_id, _ = os.path.splitext(tempfilename)
        vid_features = torch.from_numpy(load_np(vid_path)).cuda() #(1, 512)
        if int(vid_id) < len(coco_ids):
            sents_ids_1 = sents_dir[vid_id]
            logits_per_vid = vid_features @ text_features[sents_ids_1].t() #(1, 31)
            logits_per_vid = torch.squeeze(logits_per_vid, dim=0)
            sents_ids_2 = torch.argsort(logits_per_vid, dim=-1, descending=True)[:top_k].cpu().numpy()
            sents_ids_1 = sents_ids_1[sents_ids_2]
            logits_per_vid = logits_per_vid.cpu().numpy()
            retrieval_res[vid_id] = (sents_ids_1, logits_per_vid[sents_ids_2])

        else:
            logits_per_vid = vid_features @ text_features.t() #(1，46416)
            logits_per_vid = torch.squeeze(logits_per_vid) #(46416)
            sents_ids = torch.argsort(logits_per_vid, dim=-1, descending=True)[:top_k].cpu().numpy() #(20)
            logits_per_vid = logits_per_vid.cpu().numpy() #(46416)
            retrieval_res[vid_id] = (sents_ids, logits_per_vid[sents_ids])

    print(len(retrieval_res))
    save_pickle(retrieval_res, '/video-text_retrieval/data/vatex_video2text_retrieval.pkl')
    return retrieval_res

def sent2objid(params, full_sents, retrieval_res):
    
    top_k = params['sent2obj_top_k']

    sents = [line.strip() for line in full_sents]
        
    vocab = json.load(open('/video-text_retrieval/data/semantics_labels.json', 'r'))

    obj_dict = {w:i for i,w in enumerate(vocab)}

    retrieval_objs_res = {}
    data = retrieval_res
    for image_id in tqdm(data):
        wlist = []
        sents_ids = data[image_id][0][:top_k] #选择前5个描述语句id
        sents_scores = data[image_id][1][:top_k] #选择前5个描述语句得分
        for sentid, score in zip(sents_ids, sents_scores):
            if score > 1.0:
                sent = sents[sentid]
                words = sent.split(' ')
                objs = [obj_dict[w] for w in words if w in obj_dict] #将语句中的目标单词选出
                if objs == []:
                    objs = [0]
                wlist += objs
        wlist = list(set(wlist))
        retrieval_objs_res[image_id] = wlist

    save_pickle(retrieval_objs_res, '/video-text_retrieval/data/vatex_image2text_retrieval_objs.pkl')
    return retrieval_objs_res

def clip_attr(params):
    stages = ['train', 'val', 'test']
    raw_anno = utils.load_json(os.path.join(params['anno_dir'], 'dataset.json'))['images']
    vocab = utils.load_json('/video-text_retrieval/data/semantics_labels.json')
    for stage in stages:
        obj_dict = {w:i for i,w in enumerate(vocab)}
        res = []
        for ent in tqdm(raw_anno):
            image_id = ent['imgid']
            if ent['split'] != stage:
                continue

            labels = np.zeros((len(vocab), )).astype(np.int32) #906

            gt_objs = []
            for sent in ent['sentences']:
                gto = []
                for w in sent['tokens']:
                    if w in obj_dict:
                        labels[obj_dict[w]] = 1 #将该目标单词位置上的元素设置为1
                        gto.append(obj_dict[w]) #将该目标单词的类别添加到列表
                if len(gto) > 0:
                    gt_objs.append(set(gto))
                else:
                    gt_objs.append(set([0]))

            res.append({
                'image_id': image_id, # (42)
                'labels': labels, # (1000) 包含图片对应的5个描述语句中所有的语义单词
                'gt_objs': gt_objs, #[{288, 582, 202, 618, 666, 62}, {618, 670, 343, 23}, {288, 850, 43, 502}, {23, 12, 582, 15}, {618, 343, 582, 23}]
                "filename": ent["filename"],
                # "filepath": ent["filepath"]
            })

        print(len(res))
        utils.save_pickle(res, '/video-text_retrieval/data/vatex_attr1000_labels_' + stage + '.pkl')

def filter(params, retrieval_objs_res):
    stages = ['train', 'val', 'test']

    for stage in stages:
        data = utils.load_pickle(os.path.join(params['anno_dir'], 'vatex_caption_anno_' + stage + '.pkl'))
        #包含图片对应的所有正确的语义单词标签
        data2 = utils.load_pickle('/video-text_retrieval/data/vatex_attr1000_labels_' + stage + '.pkl')
        #包含由Clip选择得到的每个图片对应的前5个语句中的语义单词
        retrieval_res = retrieval_objs_res
        # clip_probs = utils.load_pickle('/home/wangtao/video_caption/xmodaler-master/configs/image_caption/cosnet/COS-Net-preprocess_v1.0/data/mil_clip_coco_scores.pkl')

        count = 0
        avg_iou = 0.0
        avg_cor = 0.0
        avg_hit = 0.0
        vocab = utils.load_json('/video-text_retrieval/data/semantics_labels.json')
        avg_missing = 0
        avg_clip_len = 0

        res = []
        for ent, ent2 in tqdm(zip(data, data2)):
            count = count + 1
            image_id = ent['video_id'] #0
            full_gt_objs = set(np.where(ent2['labels'] > 0)[0]) # {9, 74, 19, 39} 该图片包含的所有的正确语义词
            gt_objs = ent2['gt_objs'] #[{9, 19}, {9, 74, 19, 39}]

            # clip_prob = clip_probs[image_id] #(906)
            clip_objs= retrieval_res[image_id] #[9, 74, 19, 39]
        
            # clip_objs, objs_scores = filter(clip_prob, ret_objs) #[35, 422, 102, 143, 630, 890, 191] [0.9581407, 0.8375903, 0.8435399, 0.8389604, 0.80499345, 0.87753016, 0.9155463]

            correct_rate = len(full_gt_objs.intersection(clip_objs)) * 1.0 / (len(clip_objs) + 1e-10) #0.99
            hit_rate = max([ len(e.intersection(clip_objs)) * 1.0 / len(e) for e in gt_objs ]) #1.0
            iou_rate = max([ len(e.intersection(clip_objs)) * 1.0 / len(e.union(clip_objs)) for e in gt_objs ] ) #0.5
            
            avg_iou += iou_rate
            avg_cor += correct_rate
            avg_hit += hit_rate

            clip_objs = np.array(clip_objs)      # final pred: bg -- len(vocab)
            # objs_scores = np.array(objs_scores)  
            clip_objs_labels = np.zeros((len(clip_objs), )).astype(np.int64) - 1 #[-1, -1, -1, -1, -1, -1]
            missing_labels = []
            avg_clip_len += len(clip_objs)

            for i in range(len(clip_objs)):
                if clip_objs[i] in full_gt_objs:
                    clip_objs_labels[i] = clip_objs[i] #array([ 35, 422, 102, 143, 630, 890, 191])
                else:
                    clip_objs_labels[i] = len(vocab) #由Clip选择但不包含在ground trues中的语义单词

            pred_set = set(clip_objs) #{35, 102, 422, 143, 630, 890, 191}
            for gt_attr in full_gt_objs:
                if gt_attr not in pred_set:
                    missing_labels.append(gt_attr) #缺失的语义单词 [353, 859, 359, 383, 9, 647, 123]
            avg_missing += len(missing_labels)

            tmp = {
                'video_id': image_id, #图片id
                'attr_pred': clip_objs, #由Clip选择得到的语义单词
                'attr_labels': clip_objs_labels, #由Clip选择同时包含在ground trues中的语义单词
                'missing_labels': missing_labels #包含在ground trues中但Clip没有选择的语义单词
            }
            if 'tokens_ids' in ent:
                tmp.update({'tokens_ids': ent['tokens_ids']})
            if 'target_ids' in ent:
                tmp.update({'target_ids': ent['target_ids']})
            if 'object_ids' in ent:
                tmp.update({'object_ids': ent['object_ids']})
            if 'soft_target_ids' in ent:
                tmp.update({'soft_target_ids': ent['soft_target_ids']})
            res.append(tmp)

        avg_iou /= count
        avg_cor /= count
        avg_hit /= count
        avg_clip_len /= count
        avg_missing /= count
        print('average iou: ' + str(avg_iou))
        print('average cor: ' + str(avg_cor))
        print('average hit: ' + str(avg_hit))
        print('avg missing: ' + str(avg_missing))
        print('avg_clip_len: ' + str(avg_clip_len))
        if not os.path.exists(os.path.join(params['anno_dir'], 'x-clip')):
            os.mkdir(os.path.join(params['anno_dir'], 'x-clip'))

        utils.save_pickle(res, os.path.join(params['anno_dir'], 'x-clip/vatex_caption_anno_clipfilter_' + stage + '.pkl'))

def main(params):
    # step0_vatex_sents
    full_sents = vatex_sents(params)

    # step2_txtfeats
    feat_sents = txtfeats(params, full_sents)
    
    # step3_simi
    retrieval_res = simi(params)

    # step4_sent2objid
    retrieval_objs_res = sent2objid(params, full_sents, retrieval_res)

    # step5_clip_attr
    clip_attr(params)

    # step6_filter
    filter(params, retrieval_objs_res)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # input json
    parser.add_argument('--coco_ids', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/few_supervision/train_id.txt')
    parser.add_argument('--anno_dir', default='/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video_captioning/dataset/vatex/few_supervision')
    parser.add_argument('--sents_top_k', default=5, type=int)
    parser.add_argument('--sent2obj_top_k', default=5, type=int)
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)