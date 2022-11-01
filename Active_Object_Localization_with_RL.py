use_cuda = False
import torch
import torch.nn as nn
from collections import namedtuple
import torchvision.transforms as transforms

from IPython.display import clear_output

import os
import imageio
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm

import glob
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import random

import sys
import traceback
import sys
import os
import tqdm.notebook as tq
import seaborn as sns

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

import torch.nn as nn
import torchvision
import warnings
warnings.filterwarnings("ignore")

LOAD = False
SAVE_MODEL_PATH = "./models/q_network"


use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
if use_cuda:
    criterion = nn.MSELoss().cuda()   
else:
    criterion = nn.MSELoss()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  #  numbers here need to be adjusted in future
])



batch_size = 32
PATH="./datasets/"

def read_voc_dataset(download=True, year='2007'):
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                             #CustomRotation()
                            ])
    voc_data =  torchvision.datasets.VOCDetection(PATH, year=year, image_set='train', 
                        download=download, transform=T)
    train_loader = DataLoader(voc_data,shuffle=True)

    voc_val =  torchvision.datasets.VOCDetection(PATH, year=year, image_set='val', 
                        download=download, transform=T)
    val_loader = DataLoader(voc_val,shuffle=False)

    return voc_data, voc_val

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__() # recopier toute la partie convolutionnelle
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16.eval() # to not do dropout
        self.features = list(vgg16.children())[0] 
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
    def forward(self, x):
        x = self.features(x)
        return x

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= 81 + 25088, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=9)
        )
    def forward(self, x):
        return self.classifier(x)

classes = ['cat', 'bird', 'motorbike', 'diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'dog', 'aeroplane', 'cow', 'sheep', 'sofa']

def sort_class_extract(datasets):    
    
    datasets_per_class = {}
    for j in classes:
        datasets_per_class[j] = {}

    for dataset in datasets:
        for i in dataset:
            img, target = i
            classe = target['annotation']['object'][0]["name"]
            filename = target['annotation']['filename']

            org = {}
            for j in classes:
                org[j] = []
                org[j].append(img)
            for i in range(len(target['annotation']['object'])):
                classe = target['annotation']['object'][i]["name"]
                org[classe].append(  [   target['annotation']['object'][i]["bndbox"], target['annotation']['size']   ]  )
            
            for j in classes:
                if len( org[j] ) > 1:
                    try:
                        datasets_per_class[j][filename].append(org[j])
                    except KeyError:
                        datasets_per_class[j][filename] = []
                        datasets_per_class[j][filename].append(org[j])       
    return datasets_per_class

def show_new_bdbox(image, labels, color='r', count=0):
    
    xmin, xmax, ymin, ymax = labels[0],labels[1],labels[2],labels[3]
    fig,ax = plt.subplots(1)
    ax.imshow(image.transpose(0, 2).transpose(0, 1))

    width = xmax-xmin
    height = ymax-ymin
    rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor=color,facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Iteration "+str(count))
    plt.savefig(str(count)+'.png', dpi=100)

def extract(index, loader):
    
    extracted = loader[index]
    ground_truth_boxes =[]
    for ex in extracted:
        img = ex[0]
        bndbox = ex[1][0]
        size = ex[1][1]
        xmin = ( float(bndbox['xmin']) /  float(size['width']) ) * 224
        xmax = ( float(bndbox['xmax']) /  float(size['width']) ) * 224

        ymin = ( float(bndbox['ymin']) /  float(size['height']) ) * 224
        ymax = ( float(bndbox['ymax']) /  float(size['height']) ) * 224

        ground_truth_boxes.append([xmin, xmax, ymin, ymax])
    return img, ground_truth_boxes

def voc_ap(rec, prec, voc2007=False):
    
    if voc2007:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def prec_rec_compute(bounding_boxes, gt_boxes, ovthresh):
    
    nd = len(bounding_boxes)
    npos = nd
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    d = 0

    for index in range(len(bounding_boxes)):
        box1 = bounding_boxes[index]
        box2 = gt_boxes[index][0]
        x11, x21, y11, y21 = box1[0], box1[1], box1[2], box1[3]
        x12, x22, y12, y22 = box2[0], box2[1], box2[2], box2[3]
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        #  Union(A,B) = A + B - Inter(A,B)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area
        # Calcul de IOU
        iou = inter_area / union_area

        if iou > ovthresh:
            tp[d] = 1.0
        else:            
            fp[d] = 1.0
        d += 1
        
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    return prec, rec

def compute_ap_and_recall(all_bdbox, all_gt, ovthresh):
    prec, rec = prec_rec_compute(all_bdbox, all_gt, ovthresh)
    ap = voc_ap(rec, prec, False)
    return ap, rec[-1]

def eval_stats_at_threshold( all_bdbox, all_gt, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    
    stats = {}
    for ovthresh in thresholds:
        ap, recall = compute_ap_and_recall(all_bdbox, all_gt, ovthresh)
        stats[ovthresh] = {'ap': ap, 'recall': recall}
    stats_df = pd.DataFrame.from_records(stats)*100
    return stats_df

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Keep
class Agent():
    def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False ):
        self.BATCH_SIZE = 100
        self.GAMMA = 0.900
        self.EPS = 1
        self.TARGET_UPDATE = 1
        self.save_path = SAVE_MODEL_PATH
        screen_height, screen_width = 224, 224
        self.n_actions = 9
        self.classe = classe

        self.feature_extractor = FeatureExtractor()
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions)
        else:
            self.policy_net = self.load_network()
            
        self.target_net = DQN(screen_height, screen_width, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.feature_extractor.eval()
        if use_cuda:
          self.feature_extractor = self.feature_extractor.cuda()
          self.target_net = self.target_net.cuda()
          self.policy_net = self.policy_net.cuda()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        
        self.alpha = alpha # €[0, 1]  Scaling factor
        self.nu = nu # Reward of Trigger
        self.threshold = threshold
        self.actions_history = []
        self.num_episodes = num_episodes
        self.actions_history += [[100]*9]*20

    # Keep
    def save_network(self):
        torch.save(self.policy_net, self.save_path+"_"+self.classe)
        print('Saved')

    # Keep
    def load_network(self):
        if not use_cuda:
            return torch.load(self.save_path+"_"+self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path+"_"+self.classe)


    # Keep
    def intersection_over_union(self, box1, box2):
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    # Keep
    def compute_reward(self, actual_state, previous_state, ground_truth):
        res = self.intersection_over_union(actual_state, ground_truth) - self.intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            return -1
        return 1
      
    # Keep
    def rewrap(self, coord):
        return min(max(coord,0), 224)
      
    # Keep
    def compute_trigger_reward(self, actual_state, ground_truth):
        res = self.intersection_over_union(actual_state, ground_truth)
        if res>=self.threshold:
            return self.nu
        return -1*self.nu

    # Keep
    def get_best_next_action(self, actions, ground_truth):
        max_reward = -99
        best_action = -99
        positive_actions = []
        negative_actions = []
        actual_equivalent_coord = self.calculate_position_box(actions)
        for i in range(0, 9):
            copy_actions = actions.copy()
            copy_actions.append(i)
            new_equivalent_coord = self.calculate_position_box(copy_actions)
            if i!=0:
                reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)
            else:
                reward = self.compute_trigger_reward(new_equivalent_coord,  ground_truth)
            
            if reward>=0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions)==0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)

    # Keep
    def select_action(self, state, actions, ground_truth):
        sample = random.random()
        eps_threshold = self.EPS
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                  return action.cpu().numpy()[0]
                except:
                  return action.cpu().numpy()
        else:
            #return np.random.randint(0,9)   # Avant implémentation d'agent expert
            return self.get_best_next_action(actions, ground_truth) # Appel à l'agent expert.

    # Keep
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Séparation des différents éléments contenus dans les différents echantillons
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = Variable(torch.cat(next_states), 
                                         volatile=True).type(Tensor)
        
        state_batch = Variable(torch.cat(batch.state)).type(Tensor)
        if use_cuda:
            state_batch = state_batch.cuda()
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)


        # Passage des états par le Q-Network ( en calculate Q(s_t, a) ) et on récupére les actions sélectionnées
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Calcul de V(s_{t+1}) pour les prochain états.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1).type(Tensor)) 

        if use_cuda:
            non_final_next_states = non_final_next_states.cuda()
        
        # Appel au second Q-Network ( celui de copie pour garantir la stabilité de l'apprentissage )
        d = self.target_net(non_final_next_states) 
        next_state_values[non_final_mask] = d.max(1)[0].view(-1,1)
        next_state_values.volatile = False

        # On calcule les valeurs de fonctions Q attendues ( en faisant appel aux récompenses attribuées )
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Calcul de la loss
        loss = criterion(state_action_values, expected_state_action_values)

        # Rétro-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    # Keep
    def compose_state(self, image, dtype=FloatTensor):
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        #print("image feature : "+str(image_feature.shape))
        history_flatten = self.actions_history.view(1,-1).type(dtype)
        state = torch.cat((image_feature, history_flatten), 1)
        return state
    
    # Keep
    def get_features(self, image, dtype=FloatTensor):
        global transform
        #image = transform(image)
        image = image.view(1,*image.shape)
        image = Variable(image).type(dtype)
        if use_cuda:
            image = image.cuda()
        feature = self.feature_extractor(image)
        #print("Feature shape : "+str(feature.shape))
        return feature.data

    # Keep
    def update_history(self, action):
        action_vector = torch.zeros(9)
        action_vector[action] = 1
        size_history_vector = len(torch.nonzero(self.actions_history))
        if size_history_vector < 9:
            self.actions_history[size_history_vector][action] = 1
        else:
            for i in range(8,0,-1):
                self.actions_history[i][:] = self.actions_history[i-1][:]
            self.actions_history[0][:] = action_vector[:] 
        return self.actions_history

    # Keep
    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):
        alpha_h = self.alpha * (  ymax - ymin )
        alpha_w = self.alpha * (  xmax - xmin )
        real_x_min, real_x_max, real_y_min, real_y_max = 0, 224, 0, 224

        for r in actions:
            if r == 1: # Right
                real_x_min += alpha_w
                real_x_max += alpha_w
            if r == 2: # Left
                real_x_min -= alpha_w
                real_x_max -= alpha_w
            if r == 3: # Up 
                real_y_min -= alpha_h
                real_y_max -= alpha_h
            if r == 4: # Down
                real_y_min += alpha_h
                real_y_max += alpha_h
            if r == 5: # Bigger
                real_y_min -= alpha_h
                real_y_max += alpha_h
                real_x_min -= alpha_w
                real_x_max += alpha_w
            if r == 6: # Smaller
                real_y_min += alpha_h
                real_y_max -= alpha_h
                real_x_min += alpha_w
                real_x_max -= alpha_w
            if r == 7: # Fatter
                real_y_min += alpha_h
                real_y_max -= alpha_h
            if r == 8: # Taller
                real_x_min += alpha_w
                real_x_max -= alpha_w
        real_x_min, real_x_max, real_y_min, real_y_max = self.rewrap(real_x_min), self.rewrap(real_x_max), self.rewrap(real_y_min), self.rewrap(real_y_max)
        return [real_x_min, real_x_max, real_y_min, real_y_max]

    # Keep
    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates ):
        max_iou = False
        max_gt = []
        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt


    # Keep
    def train(self, train_loader):
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        for i_episode in range(self.num_episodes):
            print("Episode "+str(i_episode))
            for key, value in  train_loader.items():
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                all_actions = []
        
                # Initialize the environment and state
                self.actions_history = torch.ones((9,9))
                state = self.compose_state(image)
                original_coordinates = [xmin, xmax, ymin, ymax]
                new_image = image
                done = False
                t = 0
                actual_equivalent_coord = original_coordinates
                new_equivalent_coord = original_coordinates
                while not done:
                    t += 1
                    action = self.select_action(state, all_actions, ground_truth)
                    all_actions.append(action)
                    if action == 0:
                        next_state = None
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_trigger_reward(new_equivalent_coord,  closest_gt)
                        done = True

                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(all_actions)
                        
                        new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        

                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox( ground_truth_boxes, new_equivalent_coord )
                        reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, closest_gt)
                        
                        actual_equivalent_coord = new_equivalent_coord
                    if t == 20:
                        done = True
                    self.memory.push(state, int(action), next_state, reward)

                    # Move to the next state
                    state = next_state
                    image = new_image
                    # Perform one step of the optimization (on the target network)
                    self.optimize_model()
                    
            
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if i_episode<5:
                self.EPS -= 0.18
            self.save_network()

            print('Save Complete')

LOAD = False
SAVE_MODEL_PATH = "./models"
batch_size = 32
PATH="./datasets/"

train_loader2007, val_loader2007 = read_voc_dataset(download=True, year='2007')

classes = [ 'cat', 'dog', 'bird', 'motorbike','diningtable', 'train', 'tvmonitor', 'bus', 'horse', 'car', 'pottedplant', 'person', 'chair', 'boat', 'bottle', 'bicycle', 'aeroplane', 'cow', 'sheep', 'sofa']


agents_per_class = {}
datasets_per_class = sort_class_extract([train_loader2007])
datasets_eval_per_class = sort_class_extract([val_loader2007])

for i in range(len(classes)):
    classe = classes[i]
    print("Class --- "+str(classe)+"...")
    agents_per_class[classe] = Agent(classe, alpha=0.2, num_episodes=15, load=False)
    agents_per_class[classe].train(datasets_per_class[classe])
    torch.cuda.empty_cache()