import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import argparse
import av
import torch
import numpy as np
import glob
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
from tqdm import tqdm
np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0] #172
    end_index = indices[-1] #179
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    for rate in range(frame_sample_rate, 1, -2):
        if clip_len * rate < seg_len:
            frame_sample_rate = rate
            break
    if (clip_len * rate >= seg_len):
        frame_sample_rate = 1
    converted_len = int(clip_len * frame_sample_rate) #8
    if (converted_len == 8):
        end_idx = 8
    else:
        end_idx = np.random.randint(converted_len, seg_len) #180
    start_idx = end_idx - converted_len #172
    indices = np.linspace(start_idx, end_idx, num=clip_len) #(8)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64) #array([172, 173, 174, 175, 176, 177, 178, 179])
    return indices

def count_frames(container, input_file):
    # container = av.open(input_file)
    frames = []
    try:
        for i, frame in enumerate(container.decode(video=0)):
            frames.append(frame)
    except av.error.InvalidDataError:
        print(input_file)
    
    return len(frames)

def extract_dataset_videos_embeddings(opt):
    
    save_dir_path = opt.save_path
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    all_videos_path = glob.glob(opt.videos_path + '/*.mp4')

    processor = AutoProcessor.from_pretrained("/home/wangtao/video_caption/Transformer/X-Clip/model")

    model = AutoModel.from_pretrained("/home/wangtao/video_caption/Transformer/X-Clip/model").to("cuda")

    for vid_path in tqdm(all_videos_path):
        _, tempfilename = os.path.split(vid_path)
        vid, _ = os.path.splitext(tempfilename)
        # vid = vid_path.split('/')[-1][:-5]
        save_vid_path = os.path.join(save_dir_path, vid+'.npy')

        if os.path.exists(save_vid_path):
            print('vid:{} done'.format(vid))
            continue
        
        print(vid)

        container = av.open(vid_path)

        if container.duration == 0:
            print('failed video: {}'.format(vid))
            continue

        frames_len = count_frames(container, vid)
        
        if frames_len < 8:
            print('failed video: {}'.format(vid))
            continue

        # sample 8 frames
        indices = sample_frame_indices(clip_len=8, frame_sample_rate=8, seg_len=frames_len) #array([172, 173, 174, 175, 176, 177, 178, 179])
        # indices = sample_frame_indices(clip_len=8, frame_sample_rate=8, seg_len = len(container.decode(video=0)))
        video = read_video_pyav(container, indices) #(8, 360, 640, 3)

        inputs = processor(videos=list(video), return_tensors="pt").to("cuda") #(1, 8, 3, 244,244)

        video_features = model.get_video_features(**inputs).detach().cpu().numpy()
        
        np.save(save_vid_path, video_features)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--videos_path', type=str, default="/data16t/wangtao/dataset/VATEX/Videos/")
    parse.add_argument('--save_path', type=str, default="/home/wangtao/video_caption/Few_surpervised_vedio_captioning/video-text_retrieval/data/vatex/x-clip", help='the path to save reformat files')
    opt = parse.parse_args()
    extract_dataset_videos_embeddings(opt)
