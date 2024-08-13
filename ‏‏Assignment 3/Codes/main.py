#import
import os
import sys

import pretty_midi
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#import csv

import re

import matplotlib.pyplot as plt
import gensim
from torchtext.vocab import GloVe


## Preprocessing
def preprocessing(path, chars2remove):
    """
    Description:
    This function  doing pre processing to the data, removing characters and replacing phrases which are freqent.
    :param path: the path to the data-csv file
    :param chars2remove: characters you wish to remove from the lyrics
    :return:Cleand and processed data with additional colum of midi file name
    """
    data = pd.read_csv(path, header=None)
    data = data[[0, 1, 2]]
    data.columns = ['Singer', 'Song name', 'Lyrics']

    # Convert to title case and replace spaces with underscores
    data['Singer'] = data['Singer'].apply(lambda x: x.title().replace(' ', '_'))
    data['Song name'] = data['Song name'].apply(lambda x: x.title().replace(' ', '_'))

    # Create MIDI file names column
    data['File Name'] = data['Singer'] + '_-_' + data['Song name'] + '.mid'

    # Function to remove specific characters using replace
    def remove_specific_chars(text):
        for char in chars2remove:
            text = text.replace(char, '')
        return text

    # Apply the function to the 'Lyrics' column
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace('&', '\n'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("i'm", 'i am'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("he's", 'he is'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("she's", 'she is'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("it's", 'it is'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("you're", 'you are'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("they're", 'they are'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("we're", 'we are'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("can't", 'can not'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("isn't", 'is not'))
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.replace("aren't", 'are not'))
    data['Lyrics'] = data['Lyrics'].apply(remove_specific_chars)
    data['Lyrics'] = data['Lyrics'].apply(lambda x: x.split())
    return data


class Song_Lyrics_Dataset(Dataset):
    """
    This class knows how to extract the data, in order to use it to train.
    """
    def __init__(self, path_for_midi_files, data_frame, Glove_W2V, w2i, i2w, mode_1=True):
        self.path = path_for_midi_files
        self.df = data_frame
        self.W2V = Glove_W2V
        self.W2I = w2i
        self.I2W = i2w
        self.mode_1 = mode_1

    def __len__(self):
        return self.df.shape[0]  # number of songs

    def __getitem__(self, idx):
        Lyrics_embedding = []
        Word_ind_label = []
        Lyrics, File_Name = self.df.iloc[idx][['Lyrics', 'File Name']]

        for word in Lyrics:
            Word_ind_label.append(self.W2I[word])  # index of each word
            if word in self.W2V:
                Lyrics_embedding.append(self.W2V[word])
            else:
                Lyrics_embedding.append(np.zeros(300))  # Assuming 300-dim embeddings

        Lyrics_embedding = np.array(Lyrics_embedding, dtype=np.float32)
        Lyrics_embedding = torch.tensor(Lyrics_embedding, dtype=torch.float32)  # Convert to tensor
        Lyrics_embedding = Lyrics_embedding.view(-1, 300)  # Reshape to (sequence_length, 300)
        Word_labels = torch.tensor(Word_ind_label, dtype=torch.int64)  # shape: (sequence_length,)
        if self.mode_1 == True:
            Song_Features = self.get_mode1(self.path, File_Name)
        else:
            Song_Features = self.get_mode2(self.path, File_Name)

        Song_Features = Song_Features.repeat(Lyrics_embedding.size(0), 1)
        data = torch.cat([Lyrics_embedding, Song_Features], dim=-1)

        # Shift data to predict the next word
        data = data[:-1]
        Word_labels = Word_labels[1:]
        return data, Word_labels

    def get_mode1(self, path, File_Name):
        """
        Description: This function extract features acordding to mode 1: general song's features
        :param path: path to the midi file
        :param File_Name: the related name of the song
        :return: features of the song of mode 1
        """
        try:
            song_midi = pretty_midi.PrettyMIDI(os.path.join(path, File_Name))
            tempo = np.array([song_midi.estimate_tempo()])  # size 1

            tempi = song_midi.estimate_tempi()
            min_tempi = np.array([np.min(tempi)])  # size 1
            max_tempi = np.array([np.max(tempi)])  # size 1
            mean_tempi = np.array([np.mean(tempi)])  # size 1
            std_tempi = np.array([np.std(tempi)])  # size 1
            chroma = song_midi.get_chroma().mean(-1)  # size 12
            min_chroma = np.array([np.min(chroma)])  # size 1
            max_chroma = np.array([np.max(chroma)])  # size 1
            mean_chroma = np.array([np.mean(chroma)])  # size 1
            std_chroma = np.array([np.std(chroma)])  # size 1
            beats = song_midi.get_beats(0.0)
            mean_beats_duration = np.array([np.mean(np.diff(beats))])  # size 1
            Num_beats = np.array([len(beats)])  # size 1

            # together the size is 23
            features = np.concatenate(
                [tempo, min_tempi, max_tempi, mean_tempi, std_tempi, chroma, min_chroma, max_chroma, mean_chroma,
                 std_chroma, Num_beats, mean_beats_duration])

            features = torch.from_numpy(features).float()
        except Exception as e:
            features = torch.zeros((23,), dtype=torch.float32)
        return features

    def get_mode2(self, path, File_Name):
        """

        Description: This function extract features acordding to mode 2: instruments' features
        :param path: path to the midi file
        :param File_Name: the related name of the song
        :return: features of the song of mode 2
        """
        try:
            song_midi = pretty_midi.PrettyMIDI(os.path.join(path, File_Name))

            num_instruments = len(song_midi.instruments)  # size 1
            all_pitches = []
            all_velocities = []
            all_control_numbers = []
            all_control_values = []

            for instrument in song_midi.instruments:
                if instrument.notes:
                    all_pitches.extend([note.pitch for note in instrument.notes])
                    all_velocities.extend([note.velocity for note in instrument.notes])
                if instrument.control_changes:
                    all_control_numbers.extend([cc.number for cc in instrument.control_changes])
                    all_control_values.extend([cc.value for cc in instrument.control_changes])

            # Compute statistics
            pitch_stats = self.compute(np.array(all_pitches))  # size 4
            velocity_stats = self.compute(np.array(all_velocities))  # size 4
            control_num_stats = self.compute(np.array(all_control_numbers))  # size 4
            control_val_stats = self.compute(np.array(all_control_values))  # size 4

            piano_roll = song_midi.get_piano_roll().mean(-1)  # size 128

            features = np.concatenate([np.array([num_instruments]), pitch_stats, velocity_stats, control_num_stats, control_val_stats,piano_roll])
            features = torch.from_numpy(features).float()

        except Exception as e:
            # If an error occurs, create a zero array of the expected feature size
            features = torch.zeros((145,), dtype=torch.float32)

        return features

    def compute(self, data):
        if data.size > 0:
            return np.array([np.mean(data), np.min(data), np.max(data), np.std(data)])
        else:
            return np.zeros(4)


class RNN(nn.Module):
    def __init__(self, input_size, h_layer_size, V_size):
        super(RNN, self).__init__()
        self.gru1 = nn.GRU(input_size, h_layer_size, batch_first=True)
        self.gru2 = nn.GRU(h_layer_size, h_layer_size, batch_first=True)
        self.linear1 = nn.Linear(h_layer_size, h_layer_size)
        self.dropout = nn.Dropout(p=0.5)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(h_layer_size, V_size)

    def forward(self, input, hidden=None, test_mode=False):
        input = input.to(torch.float32)
        output1, hidden1 = self.gru1(input, hidden)
        output2, hidden2 = self.gru2(output1, hidden1)
        output = self.relu(self.linear1(output2))
        output_linear = self.dropout(output)
        logits = self.linear2(output_linear)

        if test_mode:
            return logits, hidden2
        else:
            return logits


def train_model(train_dataloader, model, optimizer, criterion, num_epochs, validation_dataloader, writer):
    """
    Description: This function running the trainning process on both the train and validation set
    :param train_dataloader: the data
    :param model:model
    :param optimizer: chosen optimizer which is adam
    :param criterion: criterion-Cross entropy
    :param num_epochs: the number of epochs
    :param validation_dataloader: validation set
    :param writer: to write to tensorboard
    :return: the predicted model
    """
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        optimizer.zero_grad()

        for i, (lyrics, labels) in enumerate(train_dataloader):
            lyrics = lyrics.to(device)
            labels = labels.to(device)
            _, seq_len, _ = lyrics.size()
            h = None  # Initialize hidden state
            optimizer.zero_grad()

            input = lyrics[:, 0, :].unsqueeze(1)
            loss = 0

            for t in range(0, seq_len - 1):
                pred, h = model(input, h, test_mode=True)

                # Compute loss for the prediction at time step t
                # Use labels at time step t (next word)
                loss += criterion(pred.view(-1, pred.size(-1)), labels[:, t])

                # Always use teacher forcing: provide the true next word as input
                input = lyrics[:, t + 1, :].unsqueeze(1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() / seq_len

            if i % 100 == 0:
                with torch.no_grad():
                    model.eval()
                    loss_validation_list = []
                    for j, (val_lyrics, val_labels) in enumerate(validation_dataloader):
                        val_lyrics = val_lyrics.to(device)
                        val_labels = val_labels.to(device)
                        val_input = val_lyrics[:, 0, :].unsqueeze(1)
                        val_loss = 0
                        h_val = None

                        for t in range(0, val_lyrics.size(1) - 1):
                            val_pred, h_val = model(val_input, h_val, test_mode=True)
                            val_loss += criterion(val_pred.view(-1, val_pred.size(-1)), val_labels[:, t])
                            val_input = val_lyrics[:, t + 1, :].unsqueeze(1)

                        loss_validation_list.append(val_loss.item() / val_lyrics.size(1))

                    loss_validation = sum(loss_validation_list) / len(loss_validation_list)

                print(f'Number of songs {i}, Train Loss: {train_loss / (i + 1)}')
                print(f'Number of songs {i}, Validation Loss: {loss_validation}')

                torch.save(model.state_dict(), 'C:/Users/idogu/deep/work 3/model_state_dict.pth')
                writer.add_scalars('Train and Validation Loss',
                                   {'Train_loss': train_loss / (i + 1), 'Validation_loss': loss_validation},
                                   epoch * len(train_dataloader) + i)

        print(f'Epoch {epoch + 1} completed')

    torch.save(model.state_dict(), 'C:/Users/idogu/deep/work 3/model_state_dict.pth')

    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Parameters
characters_to_remove = ['&', '%', '\t', '-', '!', ';', '{', ']', '#', '"', '\\', ')', '(', '!', '?', '}', '.', '%',
                        '##', '[', '{', '[', ':', '/', 'ã']
path_midi_files = 'C:/Users/idogu/deep/work 3/midi_files/'
path_train = 'C:/Users/idogu/deep/work 3/lyrics_train_set.csv'
path_test = 'C:/Users/idogu/deep/work 3/lyrics_test_set.csv'
save_file_path = 'C:/Users/idogu/deep/work 3'

# imoprting the embedding
glove = GloVe(name="6B", dim=300)
# Preprocessing the training data
data = preprocessing(path_train, characters_to_remove)
# Creating our own vocabulary acordding to all the words in the different songs:
V = set()  # V= vocabulary.
for lyrics in data['Lyrics'].tolist():
    V |= set(lyrics)
word2index = {}
index2word = {}
for i, w in enumerate(V):
    word2index[w] = i
    index2word[i] = w

train, validation = train_test_split(data, test_size=0.15, shuffle=True)

train_dataset = Song_Lyrics_Dataset(path_midi_files, train, glove, word2index, index2word, mode_1=True)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_dataset = Song_Lyrics_Dataset(path_midi_files, validation, glove, word2index, index2word, mode_1=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
"""
# Mode 1 parameters

input_size = 323 # 300 (GloVe) + 23(MIDI features) mode 1
h_layer_size = 128
vocab_size = len(word2index)
num_epochs = 7

model = RNN(input_size, h_layer_size, vocab_size).to(device)
log_dir = r'C:/Users/idogu/deep/work 3/venv/runs'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir, comment='First Mode')

# Close the writer

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

#trained_model = train_model(train_dataloader, model, optimizer, criterion, num_epochs,validation_dataloader,writer)
writer.close()

"""

# mode 2
input_size = 445 # 300 (GloVe) + 145(MIDI features) mode 2
h_layer_size = 128
vocab_size = len(word2index)
num_epochs = 15

model = RNN(input_size, h_layer_size, vocab_size).to(device)
log_dir = r'C:/Users/idogu/deep/work 3/venv/runs'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir, comment='Second Mode')

train_dataset = Song_Lyrics_Dataset(path_midi_files, train, glove, word2index, index2word, mode_1=False)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_dataset = Song_Lyrics_Dataset(path_midi_files, validation, glove, word2index, index2word, mode_1=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()
writer.add_graph(model,torch.ones(1,445))
#trained_model = train_model(train_dataloader, model, optimizer, criterion, num_epochs,validation_dataloader,writer)
writer.close()



# preparing the test data set

def features_mode1(path, File_Name):
    """

    Description: This function extract features acordding to mode 1: general song's features for the test.
        :param path: path to the midi file
        :param File_Name: the related name of the song
        :return: features of the song of mode 1
    """
    try:
        song_midi = pretty_midi.PrettyMIDI(os.path.join(path, File_Name))
        tempo = np.array([song_midi.estimate_tempo()])  # size 1

        tempi = song_midi.estimate_tempi()
        min_tempi = np.array([np.min(tempi)])  # size 1
        max_tempi = np.array([np.max(tempi)])  # size 1
        mean_tempi = np.array([np.mean(tempi)])  # size 1
        std_tempi = np.array([np.std(tempi)])  # size 1
        chroma = song_midi.get_chroma().mean(-1)  # size 12
        min_chroma = np.array([np.min(chroma)])  # size 1
        max_chroma = np.array([np.max(chroma)])  # size 1
        mean_chroma = np.array([np.mean(chroma)])  # size 1
        std_chroma = np.array([np.std(chroma)])  # size 1
        beats = song_midi.get_beats(0.0)
        mean_beats_duration = np.array([np.mean(np.diff(beats))])  # size 1
        Num_beats = np.array([len(beats)])  # size 1



        features = np.concatenate([tempo, min_tempi, max_tempi, mean_tempi, std_tempi, chroma, min_chroma, max_chroma, mean_chroma,std_chroma, Num_beats, mean_beats_duration])

        features = torch.from_numpy(features).float()
    except Exception as e:
        features = torch.zeros((23,), dtype=torch.float32)
    return features

def features_mode2( path, File_Name):
    """
    Description: This function extract features acordding to mode 2: instruments' features for the test.
        :param path: path to the midi file
        :param File_Name: the related name of the song 
        :return: features of the song of mode 2
    """
    try:
            song_midi = pretty_midi.PrettyMIDI(os.path.join(path, File_Name))

            num_instruments = len(song_midi.instruments)  # size 1
            all_pitches = []
            all_velocities = []
            all_control_numbers = []
            all_control_values = []

            for instrument in song_midi.instruments:
                if instrument.notes:
                    all_pitches.extend([note.pitch for note in instrument.notes])
                    all_velocities.extend([note.velocity for note in instrument.notes])
                if instrument.control_changes:
                    all_control_numbers.extend([cc.number for cc in instrument.control_changes])
                    all_control_values.extend([cc.value for cc in instrument.control_changes])

            # Compute statistics
            pitch_stats = compute(np.array(all_pitches))  # size 4
            velocity_stats = compute(np.array(all_velocities))  # size 4
            control_num_stats = compute(np.array(all_control_numbers))  # size 4
            control_val_stats = compute(np.array(all_control_values))  # size 4

            piano_roll = song_midi.get_piano_roll().mean(-1)  # size 128

            features = np.concatenate([np.array([num_instruments]), pitch_stats, velocity_stats, control_num_stats, control_val_stats,piano_roll])
            features = torch.from_numpy(features).float()

    except Exception as e:
            # If an error occurs, create a zero array of the expected feature size
            features = torch.zeros((145,), dtype=torch.float32)

    return features

def compute(data):
    if data.size > 0:
         return np.array([np.mean(data), np.min(data), np.max(data), np.std(data)])
    else:
        return np.zeros(4)



def Song_generator(embedd_f_word, first_song_features, model, len_sq, word2index, index2word, line_length,num_ran_word, temperature=2.0):
    """
    :param embedd_f_word: the embedding of the first word
    :param first_song_features: the features of the overall song, mode 1 or mode 2
    :param model: the model
    :param len_sq: max words in song
    :param word2index: word to index
    :param index2word: turninn index to word
    :param line_length: how long a line is
    :param num_ran_word: pick randomy form x number of words the next word
    :return: the generated song
"""
    input = torch.cat((embedd_f_word, first_song_features), dim=0)
    h = None
    model.eval()
    generated_sequence = []
    current_line_length = 0

    for i in range(len_sq):
        output_logits, h_new = model(input.unsqueeze(0), h, test_mode=True)

        # Apply temperature scaling
        logits = output_logits.squeeze(0) / temperature
        probabilities = F.softmax(logits, dim=-1)

        # Sample from the top k probabilities
        top_10_probabilities, top_10_indices = torch.topk(probabilities, num_ran_word)
        random_index = torch.randint(0, num_ran_word, (1,))
        sampled_value = top_10_indices[random_index]
        h=h_new


        generated_sequence.append(index2word[sampled_value.item()] + ' ')
        current_line_length += 1

        # Check if the current line length exceeds the limit
        if current_line_length >= line_length:
            generated_sequence.append('\n')
            current_line_length = 0

        Predicted_word_embedding = glove[index2word[sampled_value.item()]].clone().detach()
        input = torch.cat((Predicted_word_embedding, first_song_features), dim=0)

    return ''.join(generated_sequence)


mode_1=False
test = preprocessing(path_test, characters_to_remove)
# Change made here
song_length = 100
line_length = 7
num_ran_word = 20
h_layer_size = 128
vocab_size = len(word2index)

"""
words = ['war', 'betta', 'apple']
if mode_1:
    print('Using mode 1')
    # test using mode 1


    for i, file_name in enumerate(test['File Name']):
        for word in words:
            model = RNN(323, h_layer_size, vocab_size).to(device)
            model.load_state_dict(torch.load('C:/Users/idogu/deep/work 3/model_state_dict.pth'))
            First_word_I_embedd = glove[word].clone().detach()
            print("The word is: " + word)
            first_song_features = features_mode1(path_midi_files,file_name)
            gensong = Song_generator(First_word_I_embedd, first_song_features, model, song_length, word2index,index2word, line_length, num_ran_word,2.0)
            print('song name: ' + test['Singer'][i] + '- ' + test['Song name'][i])
            print(word +' ' + gensong)
            print('------------------------------')
else:

    # test using mode 2


    for i, file_name in enumerate(test['File Name']):
        for word in words:
            First_word_I_embedd = glove[word].clone().detach()
            print("The word is: " + word)
            first_song_features = features_mode2(path_midi_files, file_name)
            model = RNN(445, h_layer_size, vocab_size).to(device)
            model.load_state_dict(torch.load('C:/Users/idogu/deep/work 3/model_state_dict.pth'))
            gensong = Song_generator(First_word_I_embedd, first_song_features, model, song_length, word2index,index2word, line_length, num_ran_word,2.0)
            print('song name: ' + test['Singer'][i] + '- ' + test['Song name'][i])
            print(word + ' ' + gensong)
            print('------------------------------')
"""
"""
word='apple'
i=4
First_word_I_embedd = glove[word].clone().detach()
print("The word is: " + word)
first_song_features = features_mode2(path_midi_files, test['File Name'][i])
model = RNN(445, h_layer_size, vocab_size).to(device)
model.load_state_dict(torch.load('C:/Users/idogu/deep/work 3/model_state_dict.pth'))
gensong = Song_generator(First_word_I_embedd, first_song_features, model, song_length, word2index,index2word, line_length, num_ran_word,2.0)
print('song name: ' + test['Singer'][i] + '- ' + test['Song name'][i])
print(word + ' ' + gensong)
print('------------------------------')
"""



