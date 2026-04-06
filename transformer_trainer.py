#%%
from transformer import Transformer
import torch


START_TOKEN = '<start>'
PADDING_TOKEN = '<pad>'
END_TOKEN = '<end>'

kannada_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ', 
                    'ँ', 'ఆ', 'ఇ', 'ా', 'ి', 'ీ', 'ు', 'ూ', 
                    'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ', 
                    'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 
                    'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 
                    'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 
                    'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ', 
                    'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 
                    'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ', 
                    '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ', 'ೣ', 'ಂ', 'ಃ', 
                    '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', PADDING_TOKEN, END_TOKEN]

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

english_to_index = {token: index for index, token in enumerate(english_vocabulary)}
kannada_to_index = {token: index for index, token in enumerate(kannada_vocabulary)}
index_to_english = {index: token for token, index in english_to_index.items()}
index_to_kannada = {index: token for token, index in kannada_to_index.items()}

#%%

english_file = 'translation_en_kn/train.en' # replace this path with appropriate one
kannada_file = 'translation_en_kn/train.kn' # replace this path with appropriate one

with open(english_file, 'r') as file:
    english_sentences = file.readlines()
with open(kannada_file, 'r') as file:
    kannada_sentences = file.readlines()

english_sentences = english_sentences[:100000]
kannada_sentences = kannada_sentences[:100000]
english_sentences = [sentence.rstrip('\n').lower() for sentence in english_sentences]
kannada_sentences = [sentence.rstrip('\n') for sentence in kannada_sentences]


english_sentences[:10]
kannada_sentences[:10]
#%%

max_sequence_length = 200

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

valid_sentence_indicies = []
for index in range(len(kannada_sentences)):
    kannada_sentence, english_sentence = kannada_sentences[index], english_sentences[index]
    if is_valid_length(kannada_sentence, max_sequence_length) \
      and is_valid_length(english_sentence, max_sequence_length) \
      and is_valid_tokens(kannada_sentence, kannada_vocabulary):
        valid_sentence_indicies.append(index)


kannada_sentences = [kannada_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

print(f"Number of sentences: {len(kannada_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")


#%%
d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
kn_vocab_size = len(kannada_vocabulary)

transformer = Transformer(
    num_layers=num_layers,
    L=max_sequence_length,
    embed_dim=d_model,
    n_heads=num_heads,
    dropout=drop_prob,
    ff_hidden=ffn_hidden,
    english_to_index=english_to_index,
    kannada_to_index=kannada_to_index,
    START_TOKEN=START_TOKEN,
    END_TOKEN=END_TOKEN,
    PADDING_TOKEN=PADDING_TOKEN,
)


#%%

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.kannada_sentences[idx]

dataset = TextDataset(english_sentences, kannada_sentences)
val_split = int(0.9 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#%%


from torch import nn
# When computing the loss, we are ignoring cases when the label is the padding token
criterian = nn.CrossEntropyLoss(ignore_index=kannada_to_index[PADDING_TOKEN],
                                reduction='none')


for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%%
NEG_INFTY = float('-inf')

import numpy as np
def create_masks(eng_batch, kn_batch, max_seq_length): # en_padding_mask, de_padding_mask, cross_attention_mask -> B x L x L
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_seq_length, max_seq_length] , True) # L x L
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    en_padding_mask = torch.full([num_sentences, max_seq_length, max_seq_length] , False) # B x L x L
    de_padding_mask = torch.full([num_sentences, max_seq_length, max_seq_length] , False) # B x L x L
    cross_attention_mask = torch.full([num_sentences, max_seq_length, max_seq_length] , False) # B x L x L

    for idx in range(num_sentences):
      eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_seq_length)
      kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_seq_length)
      en_padding_mask[idx, :, eng_chars_to_padding_mask] = True
    #   en_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      de_padding_mask[idx, :, kn_chars_to_padding_mask] = True
    #   decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      cross_attention_mask[idx, :, eng_chars_to_padding_mask] = True
    #   decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(en_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + de_padding_mask, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(cross_attention_mask, NEG_INFTY, 0)
    # print(f"encoder_self_attention_mask {encoder_self_attention_mask.size()}: {encoder_self_attention_mask[0, :10, :10]}")
    # print(f"decoder_self_attention_mask {decoder_self_attention_mask.size()}: {decoder_self_attention_mask[0, :10, :10]}")
    # print(f"decoder_cross_attention_mask {decoder_cross_attention_mask.size()}: {decoder_cross_attention_mask[0, 0, :]}")
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


#%%
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_losses(train_losses, val_losses):
    clear_output(wait=True)
    plt.figure(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.show()

transformer.train()
transformer.to(device)
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    transformer.train()
    epoch_train_loss = 0
    epoch_train_count = 0

    for batch_num, batch in enumerate(train_loader):
        eng_batch, kn_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, kn_batch, max_sequence_length)
        optim.zero_grad()
        kn_predictions = transformer(eng_batch,
                                     kn_batch,
                                     encoder_self_attention_mask.to(device), 
                                     decoder_self_attention_mask.to(device), 
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=False)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
        loss = criterian(
            kn_predictions.reshape(-1, kn_vocab_size).to(device), # B x L x y_vocab_size
            labels.reshape(-1).to(device) # (B*L,)
        ).to(device)
        valid_indicies = torch.where(labels.reshape(-1) == kannada_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()

        epoch_train_loss += loss.item()
        epoch_train_count += 1

        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Kannada Translation: {kn_batch[0]}")
            kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in kn_sentence_predicted:
                if idx == kannada_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_kannada[idx.item()]
            print(f"Kannada Prediction: {predicted_sentence}")

            transformer.eval()
            kn_sentence = ("",)
            eng_sentence = ("should we go to the mall?",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_sentence, kn_sentence, max_seq_length=max_sequence_length)
                predictions = transformer(eng_sentence,
                                          kn_sentence,
                                          encoder_self_attention_mask.to(device), 
                                          decoder_self_attention_mask.to(device), 
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_kannada[next_token_index]
                kn_sentence = (kn_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                    break
            print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")
            print("-------------------------------------------")

    train_losses.append(epoch_train_loss / epoch_train_count)

    transformer.eval()
    epoch_val_loss = 0
    epoch_val_count = 0
    with torch.no_grad():
        for batch in val_loader:
            eng_batch, kn_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, kn_batch, max_sequence_length)
            kn_predictions = transformer(eng_batch,
                                         kn_batch,
                                         encoder_self_attention_mask.to(device),
                                         decoder_self_attention_mask.to(device),
                                         decoder_cross_attention_mask.to(device),
                                         enc_start_token=False,
                                         enc_end_token=False,
                                         dec_start_token=True,
                                         dec_end_token=False)
            labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
            loss = criterian(
                kn_predictions.reshape(-1, kn_vocab_size).to(device),
                labels.reshape(-1).to(device)
            ).to(device)
            valid_indicies = torch.where(labels.reshape(-1) == kannada_to_index[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_indicies.sum()
            epoch_val_loss += loss.item()
            epoch_val_count += 1

    val_losses.append(epoch_val_loss / epoch_val_count)
    print(f"Epoch {epoch} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
    plot_losses(train_losses, val_losses)


#%%
transformer.eval()
def translate(eng_sentence):
  eng_sentence = (eng_sentence,)
  kn_sentence = ("",)
  for word_counter in range(max_sequence_length):
    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, kn_sentence, max_seq_length=max_sequence_length)
    predictions = transformer(eng_sentence,
                              kn_sentence,
                              encoder_self_attention_mask.to(device), 
                              decoder_self_attention_mask.to(device), 
                              decoder_cross_attention_mask.to(device),
                              enc_start_token=False,
                              enc_end_token=False,
                              dec_start_token=True,
                              dec_end_token=False)
    next_token_prob_distribution = predictions[0][word_counter]
    next_token_index = torch.argmax(next_token_prob_distribution).item()
    next_token = index_to_kannada[next_token_index]
    kn_sentence = (kn_sentence[0] + next_token, )
    if next_token == END_TOKEN:
      break
  return kn_sentence[0]