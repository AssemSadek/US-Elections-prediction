data = read.csv(file.choose(), na.strings=c("", "NA"), header =  TRUE)

# remove USER_ID column 
drop = c("USER_ID")
data = data[, !(names(data) %in% drop)]

# remove YOB that are out range
data = data[(data$YOB < 1999 & data$YOB > 1910),]

# impute NA variables
library(mice)
imputed_data = complete(mice(data[names(data)]))
summary(data)
summary(imputed_data)

# save data after imputation
write.csv(imputed_data,"imputed_data.csv")

# partitioning of examples into generations
imputed_data = read.csv(file.choose(), na.strings=c("", "NA"), header =  TRUE)
imputed_data$Age = cut(imputed_data$YOB, breaks = c(1928, 1938,1948 ,1958,1968, 1978,1988,1999), labels = c("1st gen", "2nd gen", "3rd gen", "4th gen", "5th gen", "6th gen", "7th gen"), right = FALSE)
drop = c("YOB")
imputed_data = imputed_data[, !(names(imputed_data) %in% drop)]


# build logistic regression model with all features
library(caTools)
set.seed(123)
split = sample.split(imputed_data$Party, SplitRatio = 0.75)
train = subset(imputed_data, split == TRUE)
test = subset(imputed_data, split == FALSE)
model_logistic = glm(formula = Party ~ ., family = binomial(link="logit"), data = train)
model_logistic

# predicting test results
predictions = predict(model_logistic, type = 'response', newdata = test)
y_pred = ifelse(predictions > 0.5, 1, 0)

cm = table(test$Party, y_pred > 0.5)
cm
accuracy = (493 + 437)/ (493 + 230 + 196 + 437)
accuracy

# building model with significant features
form = Party ~ Income + Q118232 + Q116881 + Q116953 + Q115611 + Q115899 + Q109244 + Q106993 + Q106997 + Q101162 + Q98869
model_logistic_new = glm(form, family = binomial(link="logit"), data = train)
predictions_new = predict(model_logistic_new, type = 'response', newdata = test)
y_pred_new = ifelse(predictions > 0.5, 1, 0)

cm = table(test$Party, y_pred_new > 0.5)
cm
accuracy = (493 + 437)/ (493 + 230 + 196 + 437)
accuracy

# Reading test data
predictions_test_2016 = read.csv(file.choose(), na.strings=c("", "NA"), header =  TRUE)
party_2016 = read.csv(file.choose(), na.strings=c("", "NA"), header =  TRUE)
predictions_test_2016$Party = party_2016$Predictions

drop = c("USER_ID")
predictions_test_2016 = predictions_test_2016[, !(names(predictions_test_2016) %in% drop)]
# remove YOB that are out range
predictions_test_2016 = predictions_test_2016[(predictions_test_2016$YOB < 1999 & predictions_test_2016$YOB > 1910),]
# impute NA variables
imputed_test_data_2016 = complete(mice(predictions_test_2016[names(predictions_test_2016)], m = 3, maxit = 3))
predictions_test_2016 = predict(model_logistic_new, type = 'response', newdata = imputed_test_data_2016)
y_pred_test_2016 = ifelse(predictions_test_2016 > 0.5, 1, 0)

cm = table(imputed_test_data_2016$Party, y_pred_test_2016 > 0.5)
cm
accuracy = (621 + 461) / (621 + 461 + 120 + 145)
accuracy

# Visualization
party_table = table(imputed_test_data_2016$Party)
party_table
pie(party_table, main = "Pie Chart of Parties")


lbls = c("true democrat", "false republican", "false democrat", "true republican")
percentages = c(cm[1,1], cm[2,1], cm[1,2],  cm[2,2])
percentages = round(percentages/sum(percentages)*100)
lbls = paste(lbls, percentages)
lbls = paste(lbls,"%",sep="")
pie(cm, labels = lbls, main = "Pie Chart of predictions")


