import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import csv
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import time
from typing import Optional, Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import evaluate
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import VivitPreTrainedModel, VivitModel, AutoConfig, get_scheduler
 

def initialize_vivit_model(model_ckpt):
    # Load configuration
    config = AutoConfig.from_pretrained(model_ckpt)
    """ 
    provides a way to automatically create a configuration object for a pre-trained model. 
    The configuration object contains all the hyperparameters and settings needed to instantiate the model.
    """
    # Initialize model
    model = VivitModel.from_pretrained(model_ckpt, config=config)

    return model


class HexVivitModel(nn.Module):
    """ Model of dural modalities with 3 viewpoints """

    def __init__(self, model_north_a, model_east_a, model_west_a,  model_north_t,  model_east_t, model_west_t, num_labels, device='cuda:0'):
        super().__init__()
        self.device = torch.device(device)

        for param in model_north_a.parameters():
            param.requires_grad = False
        for param in model_east_a.parameters():
            param.requires_grad = False
        for param in model_west_a.parameters():
            param.requires_grad = False
        for param in model_north_t.parameters():
            param.requires_grad = False
        for param in model_east_t.parameters():
            param.requires_grad = False
        for param in model_west_t.parameters():
            param.requires_grad = False
        
        self.model_north_a = model_north_a.to(self.device)
        self.model_east_a = model_east_a.to(self.device)
        self.model_west_a = model_west_a.to(self.device)
        self.model_north_t = model_north_t.to(self.device)
        self.model_east_t = model_east_t.to(self.device)
        self.model_west_t = model_west_t.to(self.device)
        indv_hidden_size = self.model_north_a.config.hidden_size
       
        # Intermediate layer
        self.intermediate_layer1 = nn.Linear(indv_hidden_size*6, indv_hidden_size*12).to(self.device)
        self.intermediate_layer2 = nn.Linear(indv_hidden_size*12, indv_hidden_size*6).to(self.device)
     
        # Classification head
        self.classifier = nn.Linear(indv_hidden_size*6 , num_labels).to(self.device)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values_list, labels=None):
        pixel_values_north_a = pixel_values_list[0].to(self.device)
        pixel_values_east_a = pixel_values_list[1].to(self.device)
        pixel_values_west_a = pixel_values_list[2].to(self.device)
        pixel_values_north_t = pixel_values_list[3].to(self.device)
        pixel_values_east_t = pixel_values_list[4].to(self.device)
        pixel_values_west_t = pixel_values_list[5].to(self.device)

        with torch.no_grad():
            outputs_north_a = self.model_north_a(pixel_values=pixel_values_north_a)
            outputs_east_a = self.model_east_a(pixel_values=pixel_values_east_a)
            outputs_west_a = self.model_west_a(pixel_values=pixel_values_west_a)
            outputs_north_t = self.model_north_t(pixel_values=pixel_values_north_t)
            outputs_east_t = self.model_east_t(pixel_values=pixel_values_east_t)
            outputs_west_t = self.model_west_t(pixel_values=pixel_values_west_t)

        # Get pooled outputs
        pooled_output_north_a = outputs_north_a.pooler_output.to(self.device)
        pooled_output_east_a = outputs_east_a.pooler_output.to(self.device)
        pooled_output_west_a = outputs_west_a.pooler_output.to(self.device)
        pooled_output_north_t = outputs_north_t.pooler_output.to(self.device)
        pooled_output_east_t = outputs_east_t.pooler_output.to(self.device)
        pooled_output_west_t = outputs_west_t.pooler_output.to(self.device)

        # Concatenate pooled outputs
        pooled_outputs = torch.cat((pooled_output_north_a, pooled_output_east_a, pooled_output_west_a, 
                                    pooled_output_north_t, pooled_output_east_t, pooled_output_west_t), dim=-1)

        # Pass through intermediate layer
        intermediate_output1 = self.intermediate_layer1(pooled_outputs)
        intermediate_output2 = self.intermediate_layer2(intermediate_output1)

        gelu_output = F.gelu(intermediate_output2)
        
        # Apply the classification head
        logits = self.classifier(gelu_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


class TriVivitModel(nn.Module):
    """ Model of single modality with 3 viewpoints """

    def __init__(self, model_north, model_east, model_west, num_labels, device='cuda:0'):
        super().__init__()
        self.device = torch.device(device)

        for param in model_north.parameters():
            param.requires_grad = False
        for param in model_east.parameters():
            param.requires_grad = False
        for param in model_west.parameters():
            param.requires_grad = False
        
        self.model_north = model_north.to(self.device)
        self.model_east = model_east.to(self.device)
        self.model_west = model_west.to(self.device)
        indv_hidden_size = self.model_north.config.hidden_size

        # Intermediate layer
        self.intermediate_layer1 = nn.Linear(indv_hidden_size*3, indv_hidden_size*6).to(self.device)
        self.intermediate_layer2 = nn.Linear(indv_hidden_size*6, indv_hidden_size*3).to(self.device)
        
        # Classification head
        self.classifier = nn.Linear(indv_hidden_size*3 , num_labels).to(self.device)  
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values_list, labels=None):
        pixel_values_north = pixel_values_list[0].to(self.device)
        pixel_values_east = pixel_values_list[1].to(self.device)
        pixel_values_west = pixel_values_list[2].to(self.device)

        with torch.no_grad():
            outputs_north = self.model_north(pixel_values=pixel_values_north)
            outputs_east = self.model_east(pixel_values=pixel_values_east)
            outputs_west = self.model_west(pixel_values=pixel_values_west)

        # Get pooled outputs
        pooled_output_north = outputs_north.pooler_output.to(self.device)
        pooled_output_east = outputs_east.pooler_output.to(self.device)
        pooled_output_west = outputs_west.pooler_output.to(self.device)

        # Concatenate pooled outputs
        pooled_outputs = torch.cat((pooled_output_north, pooled_output_east, pooled_output_west), dim=-1)

        # Pass through intermediate layer
        intermediate_output1 = self.intermediate_layer1(pooled_outputs)
        intermediate_output2 = self.intermediate_layer2(intermediate_output1)

        gelu_output = F.gelu(intermediate_output2)
        
        # Apply the classification head
        logits = self.classifier(gelu_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


class VideoDataProcessor:
    """ Class to read video direcory """

    def __init__(self, df, first_frames, weight_stren, m_b_n, modality):
        self.df = df
        self.first_frames = first_frames
        self.weight_stren = weight_stren
        self.m_b_n = m_b_n
        self.modality = modality.lower()

    def videos_classes(self, modality):
        """
        Reads video data based on modality ('color', 'thermal', or '2m' for both).
        """
        
        modality_map = {
            'thermal': 'croppedthermal',
            'color': 'croppedcolor'
        }

        if self.modality not in modality_map and self.modality != "2m":
            raise ValueError(f"Unsupported modality '{self.modality}'. Use 'color', 'thermal', or '2m'.")

        if self.modality == "2m":

            # Process both color and thermal modalities
            videos_north_color, videos_east_color, videos_west_color, video_classes = self._process_modality(modality_map['color'], modality="color")
            videos_north_thermal, videos_east_thermal, videos_west_thermal, _ = self._process_modality(modality_map['thermal'], modality="thermal")
            
            # Return both modalities
            return (
                videos_north_color, videos_east_color, videos_west_color, 
                videos_north_thermal, videos_east_thermal, videos_west_thermal, video_classes
            )
        else:
            # Process the requested modality only (color or thermal)
            return self._process_modality(modality_map[modality], self.modality)

    def _process_modality(self, modality_folder, modality):
        base_path = os.path.dirname(os.path.abspath(__file__))

        videos_north = []
        videos_east = []
        videos_west = []
        video_classes = []

        modality_folder_path = os.path.join(base_path, modality_folder)
       
        if not os.path.isdir(modality_folder_path):
            raise FileNotFoundError(f"Folder not found: {modality_folder_path}")

        # Traverse each subject folder like 008b, 008nb
        for subject_folder in os.listdir(modality_folder_path):
            subject_path = os.path.join(modality_folder_path, subject_folder)
            
            if not os.path.isdir(subject_path):
                continue

            # Check for blind (b), non-blind (nb), or both (m) conditions
            # if self.m_b_n == 'b' and not subject_folder.endswith('b'):
            #     continue  # Skip non-blind folders if m_b_n is 'b'
            # elif self.m_b_n == 'n' and not subject_folder.endswith('nb'):
            #     continue  # Skip blind folders if m_b_n is 'n'
            # elif self.m_b_n == 'm' and not (subject_folder.endswith('b') or subject_folder.endswith('nb')):
            #     continue  # Skip folders that don't match either 'b' or 'nb' if m_b_n is 'm'
            if self.m_b_n == 'b' and not subject_folder[3] == 'b':
                continue
            elif self.m_b_n == 'n' and not subject_folder[3] == 'n':
                continue
            elif self.m_b_n == 'm' and not (subject_folder[3] == 'b' or subject_folder[3] == 'n'):
                continue
            
            for capture_folder in os.listdir(subject_path):
                foldername = os.path.basename(capture_folder)
                
                try:
                    matched_row = self.df[self.df['FolderName'] == foldername]
                    if not matched_row.empty:
                        if self.weight_stren.lower() == 'w':
                            capture_cls = int(matched_row.iloc[0]['Weight'])
                        elif self.weight_stren.lower() == 's':
                            capture_cls = int(matched_row.iloc[0]['Stren'])
                        else:
                            raise ValueError(f"Invalid value for weight_stren: {self.weight_stren}")
                    else:
                        raise ValueError(f"Folder '{foldername}' not found in DataFrame.")
                    
                except KeyError as e:
                    raise ValueError(f"Missing column in DataFrame: {e}")
                    
                video_classes.append(capture_cls)

                capture_path = os.path.join(subject_path, capture_folder)
                
                if not os.path.isdir(capture_path):
                    continue

                files = []
                # Iterate through the frame files (e.g., PC001color_01.jpg, PC005thermal_01.png)
                for frame_file in sorted(os.listdir(capture_path)):
                    frame_path = os.path.join(capture_path, frame_file)
                    if not (frame_file.endswith(".jpg") or frame_file.endswith(".png")):
                        raise ValueError(f"Unexpected file type: {frame_file} in {capture_path}")

                    files.append(frame_path)
                if len(files) != 96:
                    raise ValueError(f"Expected 96 frames, but found {len(files)} in {capture_path}.")
                
                north = files[:32]
                east = files[32:64]
                west = files[64:]

                if self.first_frames < 32:
                    black_num = 32 - self.first_frames
                    if modality.lower() == "color":
                        north = north[:self.first_frames] + ["BlackFrames/black_color.jpg"] * black_num
                        east = east[:self.first_frames] + ["BlackFrames/black_color.jpg"] * black_num
                        west = west[:self.first_frames] + ["BlackFrames/black_color.jpg"] * black_num

                    elif modality.lower() == "thermal":
                        north = north[:self.first_frames] + ["BlackFrames/black_thermal.png"] * black_num
                        east = east[:self.first_frames] + ["BlackFrames/black_thermal.png"] * black_num
                        west = west[:self.first_frames] + ["BlackFrames/black_thermal.png"] * black_num

                videos_north.append(north)
                videos_east.append(east)
                videos_west.append(west)

        return videos_north, videos_east, videos_west, video_classes


def split_videos_by_user_fold(*args, modality="color"):
    """
    Splits videos into folds by user ID.

    For 'color' or 'thermal':
        args = (videos_north, videos_east, videos_west, video_classes)

    For '2m':
        args = (
            videos_north_color, videos_east_color, videos_west_color,
            videos_north_thermal, videos_east_thermal, videos_west_thermal,
            video_classes
        )
    """
    folds = {
        "fold_1": ["030", "007", "017", "019", "032", "014"],
        "fold_2": ["028", "012", "010", "006", "031", "021"],
        "fold_3": ["016", "025", "011", "022", "024", "018"],
        "fold_4": ["029", "015", "026", "008", "020"]
    }

    if modality == "2m":
        (
            videos_north_color, videos_east_color, videos_west_color,
            videos_north_thermal, videos_east_thermal, videos_west_thermal,
            video_classes
        ) = args
    else:
        (videos_north, videos_east, videos_west, video_classes) = args

    fold_data = {}

    for fold_name in folds:
        fold_data[fold_name] = {
            'classes': [],
        }
        if modality == "2m":
            fold_data[fold_name].update({
                'north_color': [],
                'east_color': [],
                'west_color': [],
                'north_thermal': [],
                'east_thermal': [],
                'west_thermal': [],
            })
        else:
            fold_data[fold_name].update({
                'north': [],
                'east': [],
                'west': [],
            })

    for i, cls in enumerate(video_classes):
        # Use color north path to get subject ID regardless of modality
        if modality == "2m":
            sample_path = videos_north_color[i][0]
        else:
            sample_path = videos_north[i][0]

        parts = sample_path.split(os.sep)
        subject_id = parts[-3][:3]  # Extract "008" from e.g., "008b"

        for fold_name, user_ids in folds.items():
            if subject_id in user_ids:
                fold_data[fold_name]['classes'].append(cls)

                if modality == "2m":
                    fold_data[fold_name]['north_color'].append(videos_north_color[i])
                    fold_data[fold_name]['east_color'].append(videos_east_color[i])
                    fold_data[fold_name]['west_color'].append(videos_west_color[i])
                    fold_data[fold_name]['north_thermal'].append(videos_north_thermal[i])
                    fold_data[fold_name]['east_thermal'].append(videos_east_thermal[i])
                    fold_data[fold_name]['west_thermal'].append(videos_west_thermal[i])
                else:
                    fold_data[fold_name]['north'].append(videos_north[i])
                    fold_data[fold_name]['east'].append(videos_east[i])
                    fold_data[fold_name]['west'].append(videos_west[i])
                break

    return fold_data

def split_videos_randomly(*args, modality="color", test_size=0.5, random_state=42):
    """
    Randomly splits videos into two folds using train_test_split.

    Args:
        test_size: Proportion of data to go into fold_2.
        random_state: Seed for reproducibility.

    For 'color' or 'thermal':
        args = (videos_north, videos_east, videos_west, video_classes)

    For '2m':
        args = (
            videos_north_color, videos_east_color, videos_west_color,
            videos_north_thermal, videos_east_thermal, videos_west_thermal,
            video_classes
        )
    """
    if modality == "2m":
        (
            videos_north_color, videos_east_color, videos_west_color,
            videos_north_thermal, videos_east_thermal, videos_west_thermal,
            video_classes
        ) = args

        indices = list(range(len(video_classes)))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        fold_data = {
            'fold_1': {
                'north_color': [videos_north_color[i] for i in train_idx],
                'east_color': [videos_east_color[i] for i in train_idx],
                'west_color': [videos_west_color[i] for i in train_idx],
                'north_thermal': [videos_north_thermal[i] for i in train_idx],
                'east_thermal': [videos_east_thermal[i] for i in train_idx],
                'west_thermal': [videos_west_thermal[i] for i in train_idx],
                'classes': [video_classes[i] for i in train_idx],
            },
            'fold_2': {
                'north_color': [videos_north_color[i] for i in test_idx],
                'east_color': [videos_east_color[i] for i in test_idx],
                'west_color': [videos_west_color[i] for i in test_idx],
                'north_thermal': [videos_north_thermal[i] for i in test_idx],
                'east_thermal': [videos_east_thermal[i] for i in test_idx],
                'west_thermal': [videos_west_thermal[i] for i in test_idx],
                'classes': [video_classes[i] for i in test_idx],
            }
        }

    else:
        videos_north, videos_east, videos_west, video_classes = args

        indices = list(range(len(video_classes)))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        fold_data = {
            'fold_1': {
                'north': [videos_north[i] for i in train_idx],
                'east': [videos_east[i] for i in train_idx],
                'west': [videos_west[i] for i in train_idx],
                'classes': [video_classes[i] for i in train_idx],
            },
            'fold_2': {
                'north': [videos_north[i] for i in test_idx],
                'east': [videos_east[i] for i in test_idx],
                'west': [videos_west[i] for i in test_idx],
                'classes': [video_classes[i] for i in test_idx],
            }
        }

    return fold_data


class CustomVideoDataset(Dataset):
    def __init__(self, *video_args, modality, reference_image_path="croppedthermal/008b/210812_008l01w00b/PC001thermal_01.png"):
        self.modality = modality.lower()

        if self.modality == "2m":
            (
                self.video_data_north_color,
                self.video_data_east_color,
                self.video_data_west_color,
                self.video_data_north_thermal,
                self.video_data_east_thermal,
                self.video_data_west_thermal,
                self.video_classes
            ) = video_args
        else:
            (
                self.video_data_north,
                self.video_data_east,
                self.video_data_west,
                self.video_classes
            ) = video_args

        # Load and convert reference image to grayscale for thermal comparison (if applicable)
        if self.modality in ['thermal', '2m'] and reference_image_path:
            with Image.open(reference_image_path) as ref_img:
                self.reference_array = np.array(ref_img).astype(np.float32)
                self.ref_min = np.min(self.reference_array)
                self.ref_max = np.max(self.reference_array)
        else:
            self.reference_array = None
            self.ref_min = None
            self.ref_max = None

    def __len__(self):
        return len(self.video_classes)

    def process_frames(self, video_frames, is_thermal=False):
        frames = []
        for frame_path in video_frames:
            
            with Image.open(frame_path) as img:
                if is_thermal:
                    
                    frame_array = np.array(img).astype(np.float32)
                    frame_array = np.nan_to_num(frame_array, nan=4900.0, posinf=5600.0, neginf=4900.0)
                    
                    if self.reference_array is not None:
                        current_min = np.min(frame_array)
                        current_max = np.max(frame_array)

                        if current_max > current_min  and self.ref_max > self.ref_min:
                            frame_array = (frame_array - current_min) / (current_max - current_min)
                            frame_array = frame_array * (self.ref_max - self.ref_min) + self.ref_min
                            frame_array = (frame_array - self.ref_min) / (self.ref_max - self.ref_min)
                            frame_array = np.clip(frame_array * 255.0, 0, 255).astype(np.uint8)
                            
                        else: 
                            frame_array = np.full_like(frame_array, 127, dtype=np.uint8)
                        frame_array = np.stack([frame_array] * 3, axis=-1)

                    else: 
                        frame_array = np.stack([frame_array.astype(np.uint8)] * 3, axis=-1)
                        
                    frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
                else:
                    img_rgb = img.convert('RGB')
                    frame_array = np.array(img_rgb).astype(np.float32)
                    frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        label = torch.tensor(int(self.video_classes[idx]), dtype=torch.long)

        if self.modality == "2m":
            return {
                'pixel_values_north_color': self.process_frames(self.video_data_north_color[idx], is_thermal=False),
                'pixel_values_east_color': self.process_frames(self.video_data_east_color[idx], is_thermal=False),
                'pixel_values_west_color': self.process_frames(self.video_data_west_color[idx], is_thermal=False),
                'pixel_values_north_thermal': self.process_frames(self.video_data_north_thermal[idx], is_thermal=True),
                'pixel_values_east_thermal': self.process_frames(self.video_data_east_thermal[idx], is_thermal=True),
                'pixel_values_west_thermal': self.process_frames(self.video_data_west_thermal[idx], is_thermal=True),
                'labels': label
            }
        else:
            is_thermal = self.modality == "thermal"
            return {
                'pixel_values_north': self.process_frames(self.video_data_north[idx], is_thermal=is_thermal),
                'pixel_values_east': self.process_frames(self.video_data_east[idx], is_thermal=is_thermal),
                'pixel_values_west': self.process_frames(self.video_data_west[idx], is_thermal=is_thermal),
                'labels': label
            }


def train_model(model, train_dataloader, num_epochs, optimizer, lr_scheduler, device, fold, save_dir):
    """
    Train the model for a given fold, supporting both single and dual modality inputs.
    """
    model.train()
    for epoch in trange(num_epochs, desc=f"Training Fold {fold + 1}"):
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for step, batch in enumerate(progress_bar):
            # Determine modality based on keys
            if 'pixel_values_north_color' in batch:
                # Dual modality
                pixel_values_list = [
                    batch['pixel_values_north_color'].to(device),
                    batch['pixel_values_east_color'].to(device),
                    batch['pixel_values_west_color'].to(device),
                    batch['pixel_values_north_thermal'].to(device),
                    batch['pixel_values_east_thermal'].to(device),
                    batch['pixel_values_west_thermal'].to(device)
                ]
            else:
                pixel_values_list = [
                    batch['pixel_values_north'].to(device),
                    batch['pixel_values_east'].to(device),
                    batch['pixel_values_west'].to(device)
                ]
            labels = batch['labels'].to(device)

            # labels = torch.stack(batch['labels']).to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values_list=pixel_values_list, labels=labels)
                loss = outputs['loss']

            train_loss += loss.detach().cpu().item() / len(train_dataloader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            torch.cuda.empty_cache()
            progress_bar.update(1)

        log(f"[Fold {fold+1}] Epoch {epoch + 1}/{num_epochs} Training Loss: {train_loss:.3f}")

        # Save checkpoint
        checkpoint_path = Path(save_dir) / f"fold_{fold+1}_epoch_{epoch + 1:02d}_{train_loss:.3f}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        log(f"Saved checkpoint: {checkpoint_path}")


def test_model(model, test_dataloader, device):
    """
    Evaluate the model on the test set, supporting both single and dual modality inputs.
    """
    start_time = time.time()

    model.eval()
    all_preds = []
    all_labels = []
    prediction_time = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            
            # Check modality type by keys
            if 'pixel_values_north_color' in batch:
                # Dual modality
               pixel_values_list = [
                    batch['pixel_values_north_color'].to(device),
                    batch['pixel_values_east_color'].to(device),
                    batch['pixel_values_west_color'].to(device),
                    batch['pixel_values_north_thermal'].to(device),
                    batch['pixel_values_east_thermal'].to(device),
                    batch['pixel_values_west_thermal'].to(device)
                ]
            else:
                pixel_values_list = [
                    batch['pixel_values_north'].to(device),
                    batch['pixel_values_east'].to(device),
                    batch['pixel_values_west'].to(device)
                ]

            labels = batch['labels'].to(device)

            # with torch.amp.autocast('cuda'):
            outputs = model(pixel_values_list=pixel_values_list)
            logits = outputs['logits']

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            torch.cuda.empty_cache()
    
    end_time = time.time()
    duration = end_time - start_time

    return all_preds, all_labels, duration, len(all_labels)


def calculate_accuracy(preds, labels):
    correct = np.sum(preds == labels)
    total = len(labels)
    return correct / total * 100 


def log(message):
    print(message)


def main():
    parser = argparse.ArgumentParser(description="TriVivitModel Training")
    parser.add_argument('--first_frames', type=int, default=32, help="Number of first frames for model training.")
    parser.add_argument('--weight_stren', type=str, default='w', help="Weight or strenuousness data")
    parser.add_argument('--m_b_n', type=str, required=True, help="Data used to train model: mixed, blind, or nonblind.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the model on (e.g., 'cuda:0', 'cuda:1', or 'cpu').")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory for model saving.")
    parser.add_argument('--modality', type=str, required=True, help="Modality data for model training.")
    parser.add_argument('--train_test', type=str, default='train', help="Train or test.")
    parser.add_argument('--model_weight', nargs='*', type=str, help="A list of model weight")
    parser.add_argument('--agnostic_aware', type=str, required=True, help="Agnostic or Aware model.")
    args = parser.parse_args()

    model_ckpt = "google/vivit-b-16x2-kinetics400"
    save_dir = args.save_dir    ####
    num_labels = int(save_dir.split('/')[-1][0])
    print("***********************")
    print("save_dir: ", save_dir)
    print("num_labels: ", num_labels)
    
    save_dir_parts = save_dir.strip('/').split('/')
    num_labels = int(save_dir.split('/')[-1][0])
    save_dir_parts = save_dir.strip('/').split('/')

    if len(save_dir_parts) < 1:
        raise ValueError("Invalid save_dir format. It should contain at least one folder.")

    if save_dir_parts[1].lower() != args.modality.lower():
        raise ValueError(f"Mismatch: model would be saved in '{save_dir_parts[1]}' directory, but --modality is '{args.modality}'.")

    if save_dir_parts[-1][-1] != args.m_b_n:
        raise ValueError(f"Mismatch: model would be saved as '{save_dir_parts[-1][-1]}', but --m_b_n is '{args.m_b_n}'.")
    
    if (save_dir_parts[-1][1] == "c" and args.weight_stren.lower() == "s") or (save_dir_parts[-1][1] == "s" and args.weight_stren.lower() == "w"):
        raise ValueError(f"Mismatch: model would be saved as '{save_dir_parts[-1][1]}', but --weight_stren is '{args.weight_stren.lower()}'. ")

    if save_dir_parts[-2][-2].isnumeric():
        if save_dir_parts[-2][-2:] != str(args.first_frames):
            raise ValueError(f"Mismatch: This is a directory for first '{save_dir_parts[-2][-2:]} frames', but --first_frames ends with '{str(args.first_frames)}'.")
    else:
        if save_dir_parts[-2][-1] != str(args.first_frames):
            raise ValueError(f"Mismatch: This is a directory for first '{save_dir_parts[-2][-1]} frames', but --first_frames ends with '{str(args.first_frames)}'.")
    
    if save_dir_parts[-2][:2] != args.agnostic_aware:
        raise ValueError (f"Mismatch: model would be saved as '{save_dir_parts[-2][:2]}', but --agnostic_aware is '{args.agnostic_aware}'.")

    

    combined_csv_path = "subject_difficulty.csv"
    combined_df = pd.read_csv(combined_csv_path)     
    processor = VideoDataProcessor(combined_df, first_frames=args.first_frames, weight_stren=args.weight_stren, m_b_n=args.m_b_n, modality=args.modality)

    if args.modality.lower() == "color" or args.modality.lower() == "thermal":
        videos_north, videos_east, videos_west, video_classes = processor.videos_classes(args.modality)
    elif args.modality.lower() == "2m":
        videos_north_color, videos_east_color, videos_west_color, \
        videos_north_thermal, videos_east_thermal, videos_west_thermal, video_classes =  processor.videos_classes(modality="2m")
   
    if (args.weight_stren == 'W' or args.weight_stren == 'w') and num_labels == 3:
        video_classes = [2 if x == 3 else x for x in video_classes]  # 3-class
    elif (args.weight_stren == 'W' or args.weight_stren == 'w') and num_labels == 2:
        video_classes = [0 if x == 1 else (1 if x == 2 or x == 3 else x) for x in video_classes]  # 2-class
    elif (args.weight_stren == 'S' or args.weight_stren == 's') and num_labels == 2:
        video_classes = [0 if x == 1.0 or x == 2.0 else (1 if x == 3.0 or x == 4.0 or x == 5.0 else x) for x in video_classes]  # 2-class stren
    
    print("Weight or Strenuous: ", args.weight_stren)
    device = args.device
    print(device)
    print("***********************")

    datasets = []
    if args.modality.lower() in ["color", "thermal"]:
        if args.agnostic_aware == "ag":
            fold_data = split_videos_by_user_fold(videos_north, videos_east, videos_west, video_classes, modality=args.modality)
            
        else: 
            fold_data = split_videos_randomly(videos_north, videos_east, videos_west, video_classes, modality=args.modality)
            
        for fold_key, fold in fold_data.items():
            dataset = CustomVideoDataset(
                fold['north'],
                fold['east'],
                fold['west'],
                fold['classes'],
                modality=args.modality  # Pass modality to dataset constructor
            )
            datasets.append(dataset)

       

    else:
        if args.agnostic_aware == "ag":
            fold_data = split_videos_by_user_fold(videos_north_color, videos_east_color, videos_west_color, \
                                                videos_north_thermal, videos_east_thermal, videos_west_thermal, video_classes, modality=args.modality)
        else:
            fold_data = split_videos_randomly(videos_north_color, videos_east_color, videos_west_color, \
                                            videos_north_thermal, videos_east_thermal, videos_west_thermal, video_classes, modality=args.modality)
        for fold_key, fold in fold_data.items():
            dataset = CustomVideoDataset(
                fold['north_color'],
                fold['east_color'],
                fold['west_color'],
                fold['north_thermal'],
                fold['east_thermal'],
                fold['west_thermal'],
                fold['classes'],
                modality=args.modality  # Pass modality to dataset constructor
            )
            datasets.append(dataset)
    
    if args.train_test == "train" and not args.model_weight:

        num_epochs = 20 
        k_folds = len(datasets)
        batch_size = 4
       
        for fold in range(k_folds):
            log(f"Starting fold {fold + 1}/{k_folds}")

            # Check modality type and initialize model accordingly
            if args.modality.lower() == "thermal" or args.modality.lower()  == "color":
                # Single Modality: Use TriVivitModel
                model_north = initialize_vivit_model(model_ckpt).to(device)
                model_east = initialize_vivit_model(model_ckpt).to(device)
                model_west = initialize_vivit_model(model_ckpt).to(device)
                model = TriVivitModel(model_north, model_east, model_west, num_labels=num_labels, device=device).to(device)
            else:
                # Dual Modality: Use HexVivitModel
                model_north_color = initialize_vivit_model(model_ckpt).to(device)
                model_east_color = initialize_vivit_model(model_ckpt).to(device)
                model_west_color = initialize_vivit_model(model_ckpt).to(device)
                model_north_thermal = initialize_vivit_model(model_ckpt).to(device)
                model_east_thermal = initialize_vivit_model(model_ckpt).to(device)
                model_west_thermal = initialize_vivit_model(model_ckpt).to(device)
                model = HexVivitModel(model_north_color, model_east_color, model_west_color,
                                    model_north_thermal, model_east_thermal, model_west_thermal,
                                    num_labels=num_labels, device=device).to(device)

            # Use all data except current fold as training set
            train_datasets = [datasets[i] for i in range(k_folds) if i != fold]
            train_dataset = ConcatDataset(train_datasets)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # learning_rate = 0.01
            # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            learning_rate = 0.001
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            num_training_steps = num_epochs * len(train_dataloader)
            warmup_steps = int(0.1 * num_training_steps)
            lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )

            # Train the model for the current fold
            train_model(model, train_dataloader, num_epochs, optimizer, lr_scheduler, device, fold, save_dir)


    elif args.train_test == "test" and args.model_weight:
        all_preds_all_folds = []
        all_labels_all_folds = []
        per_sample_durations = []
        
        cm_save_dir = save_dir.replace("Results", "CM", 1)
        os.makedirs(cm_save_dir, exist_ok=True)
        
        k_folds = len(datasets)
        batch_size = 16

        for fold in range(k_folds):
            log(f"Starting fold {fold + 1}/{k_folds}")

            # Check modality type and initialize model accordingly
            if args.modality.lower() == "thermal" or args.modality.lower()  == "color":
                # Single Modality: Use TriVivitModel
                model_north = initialize_vivit_model(model_ckpt).to(device)
                model_east = initialize_vivit_model(model_ckpt).to(device)
                model_west = initialize_vivit_model(model_ckpt).to(device)
                model = TriVivitModel(model_north, model_east, model_west, num_labels=num_labels, device=device).to(device)
            else:
                # Dual Modality: Use HexVivitMode
                model_north_color = initialize_vivit_model(model_ckpt).to(device)
                model_east_color = initialize_vivit_model(model_ckpt).to(device)
                model_west_color = initialize_vivit_model(model_ckpt).to(device)
                model_north_thermal = initialize_vivit_model(model_ckpt).to(device)
                model_east_thermal = initialize_vivit_model(model_ckpt).to(device)
                model_west_thermal = initialize_vivit_model(model_ckpt).to(device)
                model = HexVivitModel(model_north_color, model_east_color, model_west_color,
                                    model_north_thermal, model_east_thermal, model_west_thermal,
                                    num_labels=num_labels, device=device).to(device)

            # Load model weights for the current fold
            parameter_path = args.model_weight[fold]
            model.load_state_dict(torch.load(parameter_path))

            # Use the current fold as the test set
            test_datasets = [datasets[fold]]

            test_dataset = ConcatDataset(test_datasets)
            
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            preds, labels, duration, num_samples = test_model(model, test_dataloader, device)
            all_preds_all_folds.extend(preds)
            all_labels_all_folds.extend(labels)

            per_sample_duration = duration / num_samples
            per_sample_durations.append(per_sample_duration)

        average_duration = np.mean(per_sample_durations)
        std_duration = np.std(per_sample_durations)
        print(f"Average prediction time per sample: {average_duration:.3f} seconds")
        print(f"Standard deviation of prediction time per sample: {std_duration:.3f} seconds")


        # Save prediction, ground truth, and the match in a CSV file
        comparison_results = [1 if p == l else 0 for p, l in zip(all_preds_all_folds, all_labels_all_folds)]
        results = list(zip(all_preds_all_folds, all_labels_all_folds, comparison_results))
        
        csv_path = os.path.join(cm_save_dir, "prediction_comparison.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Predicted', 'Actual', 'Match'])  # Header
            writer.writerows(results)

        # Plot aggregated confusion matrix
        correct_predictions = sum(comparison_results)
        total_predictions = len(comparison_results)
        average_accuracy = (correct_predictions / total_predictions) * 100
        print(f"Average Accuracy: {average_accuracy:.2f}")

        cm = confusion_matrix(all_labels_all_folds, all_preds_all_folds)


        # Set custom labels based on num_labels
        if num_labels == 4:
            custom_labels = ["00", "15", "30", "45"]
        elif num_labels == 3:
            custom_labels = ["Low", "Mid", "High"]
        else:
            custom_labels = ["Low", "High"]

        percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        percentages = np.nan_to_num(percentages)  # Replace NaNs with 0 if any class has 0 samples

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(12,10))
        ax = sns.heatmap(percentages, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=custom_labels, yticklabels=custom_labels, 
                        cbar=True, linewidths=0.5, linecolor='black', annot_kws={"size": 20})
        for text in ax.texts:
            text.set_text(f'{text.get_text()}%')
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('True Class', fontsize=14)
        plt.title('Aggregated Confusion Matrix (%)')
        plt.tight_layout()
        save_path = os.path.join(cm_save_dir, f"confusion_{average_accuracy:.2f}_{average_duration:.3f}_{std_duration:.3f}.png")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    main()


