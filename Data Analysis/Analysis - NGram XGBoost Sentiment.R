#Import required libraries
library(tidyverse)
library(stopwords)
library(text2vec)
library(data.table)
library(magrittr)
library(glmnet)
library(xgboost)
library(randomForest)


#Read in data
reviews <- read_csv("AppReview.csv")

#Select only required columns and remove unneeded data, create an index variable and 
#convert to a data table.
reviews_cleaned <- reviews %>% mutate(sentiment = round(reviewerRating)) %>% 
  select(reviewText, reviewerRating, sentiment)
reviews_cleaned$id <- 1:nrow(reviews_cleaned)
setDT(reviews_cleaned)
setkey(reviews_cleaned, id)
rm(reviews)

#Split into a training (tuning + validation) and test set
all_ids <- reviews_cleaned$id
train_ids <- sample(all_ids, nrow(reviews_cleaned)*0.8)
tune_ids <- sample(train_ids, length(train_ids)*0.8)
valid_ids <- setdiff(train_ids, tune_ids)
test_ids <- setdiff(all_ids, train_ids)
train <- reviews_cleaned[J(train_ids)]
tune <- reviews_cleaned[J(tune_ids)]
valid <- reviews_cleaned[J(valid_ids)]
test <- reviews_cleaned[J(test_ids)]

#Generate stopwords set
words_to_include <- c("no", "not", "cannot")
words_to_remove <- c("full", "review")
sw <- stopwords::stopwords("en")[!(stopwords::stopwords("en") %in% words_to_include)]
sw <- append(sw, words_to_remove)

# # --- Automatic Tuning ---
# 
# #Combinations of hyperparameters
# n_comb <- 32
# vocab_term_maxs <- c(1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
#                     2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500,
#                     5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
#                     10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000)
# tf_idfs <- c(0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L,
#              0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L)
# max_depths <- c(3L, 7L, 13L, 19L, 3L, 7L, 13L, 19L, 3L, 7L, 13L, 19L, 3L, 7L, 13L, 19L,
#                 3L, 7L, 13L, 19L, 3L, 7L, 13L, 19L, 3L, 7L, 13L, 19L, 3L, 7L, 13L, 19L)
# n_rounds <- c(1000, 400, 200, 150, 1000, 400, 200, 150,
#               1000, 400, 200, 150, 1000, 400, 200, 150,
#               1000, 400, 200, 150, 1000, 400, 200, 150,
#               1000, 400, 200, 150, 1000, 400, 200, 150)
# errors <- numeric(n_comb)
# 
# #Find best combination
# for(i in 1:n_comb){
# 
#   #Tuning iterator
#   it_tune <- itoken(tune$reviewText, preprocessor = tolower, tokenizer = word_tokenizer,
#                     ids = tune$id, progressbar = FALSE)
# 
#   #Create vocab
#   vocab = create_vocabulary(it_tune, stopwords = sw, ngram = c(1L, 3L))
#   vocab = prune_vocabulary(vocab, term_count_min = 10,
#                            doc_proportion_max = 0.65, vocab_term_max = vocab_term_maxs[i])
#   vectorizer <- vocab_vectorizer(vocab)
# 
#   #Create tuning dtm
#   dtm_tune  <- create_dtm(it_tune, vectorizer)
#   if(tf_idfs[i] == 1){
#     tfidf = TfIdf$new()
#     dtm_tune <- fit_transform(dtm_tune, tfidf)
#   }
# 
#   #Train xgboost model
#   boost_model <- xgboost(data = dtm_tune, label = tune$sentiment, max_depth = max_depths[i],
#                          eta = 0.3, nthread = 8, nrounds = n_rounds[i], objective = "binary:hinge",
#                          print_every_n = 50)
# 
#   #Build validation dtm
#   it_valid = itoken(valid$reviewText, preprocessor = tolower, tokenizer = word_tokenizer,
#                    ids = valid$id, progressbar = FALSE)
#   dtm_valid = create_dtm(it_valid, vectorizer)
#   if(tf_idfs[i] == 1){
#     dtm_valid <- transform(dtm_valid, tfidf)
#   }
# 
#   #Make predictions and calculate error
#   preds <- predict(boost_model, dtm_valid, type = 'class')
#   errors[i] <- mean(preds != valid$sentiment)
# }
# 
# # --- Manual Tuning ---
# 
# #Manually tuning best combo from automatic tuning
# 
# #Create vocab
# vocab = create_vocabulary(it_tune, stopwords = sw, ngram = c(1L, 3L))
# vocab = prune_vocabulary(vocab, term_count_min = 10,
#                          doc_proportion_max = 0.65, vocab_term_max = 10000)
# vectorizer <- vocab_vectorizer(vocab)
# 
# #Create tuning dtm
# dtm_tune  <- create_dtm(it_tune, vectorizer)
# 
# #Train xgboost model
# boost_model <- xgboost(data = dtm_tune, label = tune$sentiment, max_depth = 13,
#                        eta = 0.2, nthread = 8, nrounds = 300, objective = "binary:hinge")
# 
# #Create valid DTM
# it_valid = itoken(valid$reviewText, preprocessor = tolower, tokenizer = word_tokenizer,
#                   ids = valid$id, progressbar = FALSE)
# dtm_valid = create_dtm(it_valid, vectorizer)
# 
# #Make predictions and calculate error
# preds <- predict(boost_model, dtm_valid, type = 'class')
# mean(preds == valid$sentiment)

# --- Final Model ---

#Preprocessing and tokenization of whole training set
it_train <- itoken(train$reviewText, preprocessor = tolower, tokenizer = word_tokenizer, 
                   ids = train$id, progressbar = FALSE)

#Vocab from entire training set
vocab = create_vocabulary(it_train, stopwords = sw, ngram = c(1L, 3L))
vocab = prune_vocabulary(vocab, term_count_min = 10, 
                         doc_proportion_max = 0.65,
                         vocab_term_max = 10000)

#Vectorizer from entire training set
vectorizer <- vocab_vectorizer(vocab)

#Create training DTM
dtm_train  <- create_dtm(it_train, vectorizer)

#Train XGBoost model
boost_model <- xgboost(data = dtm_train, label = train$sentiment, max_depth = 13,
        eta = 0.3, nthread = 8, nrounds = 300, objective = "binary:logistic")

#Create test DTM
it_test <- itoken(test$reviewText, preprocessor = tolower, tokenizer = word_tokenizer, 
                  ids = train$id, progressbar = FALSE)
dtm_test <- create_dtm(it_test, vectorizer)

#Make predictions
preds_sent <- round(predict(boost_model, dtm_test))

#Calculate scores
mean(preds_sent == test$sentiment)

#--- Final Confirmation: Only Run for Generating a Boxplot of the Error Distribution ---

#Let's run stochastic holdout 25 times to see what the error distribution is like

#Let's include a stratified holdout function for future use
stratified_holdout <- function(y, epsilon){
  n <- length(y)
  labels <- unique(y)
  ind_train <- NULL
  ind_test <- NULL
  y <- sample(sample(sample(y))) #Resample y multiple times
  for(label in labels){ #Go through each label
    labels_i <- which(y == label)
    n_i <- length(labels_i)
    ind_train <- c(ind_train, sample(labels_i,
                                     round((1 - epsilon)*n_i), replace = FALSE)) #Select (1 - epsilon)% for train
  }
  ind_test <- (1:n)[-ind_train] #Everything not in training set is in test set
  return(list(ind_train, ind_test))
}

n_run <- 25
errors <- numeric(n_run)

for(i in 1:n_run){

  print(i)

  #Split into training and test sets
  ind <- stratified_holdout(reviews_cleaned$sentiment, 0.2)
  ind_train <- ind[[1]]
  ind_test <- ind[[2]]
  train <- reviews_cleaned[ind_train]
  test <- reviews_cleaned[ind_test]

  #Preprocessing and tokenization of whole training set
  it_train <- itoken(train$reviewText, preprocessor = tolower, tokenizer = word_tokenizer,
                     ids = train$id, progressbar = FALSE)

  #Vocab from entire training set
  vocab = create_vocabulary(it_train, stopwords = sw, ngram = c(1L, 3L))
  vocab = prune_vocabulary(vocab, term_count_min = 10,
                           doc_proportion_max = 0.65,
                           vocab_term_max = 10000)

  #Vectorizer from entire training set
  vectorizer <- vocab_vectorizer(vocab)

  #Create training DTM
  dtm_train  <- create_dtm(it_train, vectorizer)

  #Train XGBoost model
  boost_model <- xgboost(data = dtm_train, label = train$sentiment, max_depth = 13,
                         eta = 0.2, nthread = 8, nrounds = 300, objective = "binary:logistic",
                         print_every_n = 100)

  #Create test DTM
  it_test <- itoken(test$reviewText, preprocessor = tolower, tokenizer = word_tokenizer,
                    ids = train$id, progressbar = FALSE)
  dtm_test <- create_dtm(it_test, vectorizer)

  #Make predictions
  preds_xgb <- round(predict(boost_model, dtm_test))

  #Calculate scores
  errors[i] <- mean(preds_xgb != test$sentiment)
}

boxplot(errors, main = "Boxplot of Boosted Tree Errors", ylab = "0 - 1 Loss Accuracy")
