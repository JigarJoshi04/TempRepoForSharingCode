import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


if __name__ ==  '__main__':
          #############################################################################################################################
          ##Loading dataset
          df = pd.read_csv('reviews.csv')
          # print(df.head(5))
          # print(df.shape)
          # print(type(df['sentiment'][0]))
        #   for i in range(0,50000):
        #             df['sentiment'][i] = int(df['sentiment'][0])
          # print(type(df['sentiment'][0]))
          # print(df.info())
          print("Reading dataset is completed")

          def to_sentiment(rating):
                rating = int(rating)
                if rating <= 2:
                    return 0
                elif rating == 3:
                    return 1
                else:
                    return 2
          df['sentiment'] = df.score.apply(to_sentiment)










          #############################################################################################################################
          ##Converting sentences to tokens to ids
          PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
          tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
          sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
          tokens = tokenizer.tokenize(sample_txt)
          token_ids = tokenizer.convert_tokens_to_ids(tokens)
          print(f' Sentence: {sample_txt}')
          print(f'   Tokens: {tokens}')
          print(f'Token IDs: {token_ids}')

          encoding = tokenizer.encode_plus(
          sample_txt,
          max_length=32,
          add_special_tokens=True, # Add '[CLS]' and '[SEP]'
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',  # Return PyTorch tensors
          )
          encoding.keys()
          #next line of comment is not working rest is working
          # dict_keys(['input_ids', 'attention_mask'])
          # print(len(encoding['input_ids'][0]))
          # print(encoding['input_ids'][0])
          # print(len(encoding['attention_mask'][0]))
          # print(encoding['attention_mask'])
          # print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))

          #############################################################################################################################
          ##Creating PyTorch dataset
          MAX_LEN = 160
          class GPReviewDataset(Dataset):
                    def __init__(self, reviews, targets, tokenizer, max_len):
                              self.reviews = reviews
                              self.targets = targets
                              self.tokenizer = tokenizer
                              self.max_len = max_len
                    def __len__(self):
                              return len(self.reviews)
                    def __getitem__(self, item):
                              review = str(self.reviews[item])
                              target = self.targets[item]
                              encoding = self.tokenizer.encode_plus(
                              review,
                              add_special_tokens=True,
                              max_length=self.max_len,
                              return_token_type_ids=False,
                              pad_to_max_length=True,
                              return_attention_mask=True,
                              return_tensors='pt',
                              )
                              print("GP Review dataset is returning soemthing")
                              return {
                              'review_text': review,
                              'input_ids': encoding['input_ids'].flatten(),
                              'attention_mask': encoding['attention_mask'].flatten(),
                              'targets': torch.tensor(target, dtype=torch.long)
                              }

          df_train, df_test = train_test_split(
                    df,
                    test_size=0.7,
                    random_state=21
          )
          df_val, df_test = train_test_split(
                    df_test,
                    test_size=0.5,
                    random_state=21
          )
          print(df_train.shape, df_val.shape, df_test.shape)




          def create_data_loader(df, tokenizer, max_len, batch_size):
                    ds = GPReviewDataset(
                    reviews=df.content.to_numpy(),
                    targets=df.sentiment.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                    )
                    print("Data Loader is returning something")
                    return DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=0
                    )

          BATCH_SIZE = 32
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
          data = next(iter(train_data_loader))
          data.keys()
          print(data['input_ids'].shape)
          print(data['attention_mask'].shape)
          print(data['targets'].shape)


###############################################################
#Making model
          bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
          last_hidden_state, pooled_output = bert_model(
          input_ids=encoding['input_ids'],
          attention_mask=encoding['attention_mask']
          )
          bert_model.config.hidden_size
          print(pooled_output.shape)
          class SentimentClassifier(nn.Module):
                    def __init__(self, n_classes):
                              super(SentimentClassifier, self).__init__()
                              self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
                              self.drop = nn.Dropout(p=0.3)
                              self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
                    def forward(self, input_ids, attention_mask):
                              _, pooled_output = self.bert(
                              input_ids=input_ids,
                              attention_mask=attention_mask
                              )
                              output = self.drop(pooled_output)
                              return self.out(output)
          class_names = ['negative', 'neutral', 'positive']
          model = SentimentClassifier(len(class_names))
          # model = model.to(device)

          input_ids = data['input_ids']
          attention_mask = data['attention_mask']
          print(input_ids.shape) # batch size x seq length
          print(attention_mask.shape) # batch size x seq length
          # print(F.softmax(model(input_ids, attention_mask), dim=1))
          EPOCHS = 10
          optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
          total_steps = len(train_data_loader) * EPOCHS
          scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=0,
          num_training_steps=total_steps
          )
          loss_fn = nn.CrossEntropyLoss()


          def train_epoch(
                    model,
                    data_loader,
                    loss_fn,
                    optimizer,
                    scheduler,
                    n_examples
                    ):
                    count = 0
                    model = model.train()
                    losses = []
                    correct_predictions = 0
                    for d in data_loader:
                              print(len(data_loader))
                              print("Data is loading")
                              print("This is ",count," from the total of",len(data_loader))
                              count = count +1
                              input_ids = d["input_ids"]
                              attention_mask = d["attention_mask"]
                              targets = d["targets"]
                              outputs = model(
                              input_ids=input_ids,
                              attention_mask=attention_mask
                              )
                              _, preds = torch.max(outputs, dim=1)
                              loss = loss_fn(outputs, targets)
                              correct_predictions += torch.sum(preds == targets)
                              losses.append(loss.item())
                              loss.backward()
                              nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                              optimizer.step()
                              scheduler.step()
                              optimizer.zero_grad()
                    print('Train epochs is retruning somethins')
                    return correct_predictions.double() / n_examples, np.mean(losses)


          def eval_model(model, data_loader, loss_fn,  n_examples):
                              model = model.eval()
                              losses = []
                              correct_predictions = 0
                              count = 0
                              with torch.no_grad():
                                        for d in data_loader:
                                                  print("This is ",count," from the total of",len(data_loader))
                                                  count = count +1
                                                  input_ids = d["input_ids"]
                                                  attention_mask = d["attention_mask"]
                                                  targets = d["targets"]
                                                  outputs = model(
                                                  input_ids=input_ids,
                                                  attention_mask=attention_mask
                                                  )
                                                  _, preds = torch.max(outputs, dim=1)
                                                  loss = loss_fn(outputs, targets)
                                                  correct_predictions += torch.sum(preds == targets)
                                                  losses.append(loss.item())
                              print("Eval model is returning something")
                              return correct_predictions.double() / n_examples, np.mean(losses)


          print("Epochs are about to run")
          history = defaultdict(list)
          best_accuracy = 0
          for epoch in range(EPOCHS):
                    print("epochs are running")
                    print(f'Epoch {epoch + 1}/{EPOCHS}')
                    print('-' * 10)
                    train_acc, train_loss = train_epoch(
                    model,
                    train_data_loader,
                    loss_fn,
                    optimizer,
                    scheduler,
                    len(df_train)
                    )
                    print(f'Train loss {train_loss} accuracy {train_acc}')
                    val_acc, val_loss = eval_model(
                    model,
                    val_data_loader,
                    loss_fn,
                    len(df_val)
                    )
                    print(f'Train loss {train_loss} accuracy {train_acc}')
                    print(f'Val   loss {val_loss} accuracy {val_acc}')
                    print()
                    history['train_acc'].append(train_acc)
                    history['train_loss'].append(train_loss)
                    history['val_acc'].append(val_acc)
                    history['val_loss'].append(val_loss)
                    print(f'Train loss {train_loss} accuracy {train_acc}')
                    print(f'Val   loss {val_loss} accuracy {val_acc}')
                    print("=="*500)
                    if val_acc > best_accuracy:
                              torch.save(model.state_dict(), 'offline_model_state_004.bin')
                              best_accuracy = val_acc
                    
        #   plt.plot(history['train_acc'], label='train accuracy')
        #   plt.plot(history['val_acc'], label='validation accuracy')
        #   plt.title('Training history')
        #   plt.ylabel('Accuracy')
        #   plt.xlabel('Epoch')
        #   plt.legend()
        #   plt.ylim([0, 1])

          review_text = "Bad Very Bad"
          encoded_review = tokenizer.encode_plus(
          review_text,
          max_length=MAX_LEN,
          add_special_tokens=True,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
          )

          input_ids = encoded_review['input_ids']
          attention_mask = encoded_review['attention_mask']
          output = model(input_ids, attention_mask)
          _, prediction = torch.max(output, dim=1)
          print(f'Review text: {review_text}')
          print(f'Sentiment  : {class_names[prediction]}')