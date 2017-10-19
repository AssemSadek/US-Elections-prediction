
# coding: utf-8

# In[1]:

import graphlab


# In[2]:

data = graphlab.SFrame.read_csv('train2016.csv/')
#data = graphlab.SFrame(data)
data


# In[3]:

features = data.column_names()
features.remove('Party')
features = features[2:]
print features
initial_features = features 
target = 'Party'
print target


# In[4]:

democrat_data = data[data[target] == 'Democrat']
republican_data = data[data[target] == 'Republican']
print "Percentage of Democrat                 :", len(democrat_data) / float(len(data))
print "Percentage of Republican                :", len(republican_data) / float(len(data))
print "Total number of data :", len(data)


# In[5]:

for feature in features:
    data_one_hot_encoded = data[feature].apply(lambda x: {x: 1})    
    data_unpacked = data_one_hot_encoded.unpack(column_name_prefix = feature)
    
    # Change None's to 0's
    for column in data_unpacked.column_names():
        data_unpacked[column] = data_unpacked[column].fillna(0)

    
    data.remove_column(feature)
    columns = data_unpacked.column_names()
    if columns[0] == feature + '.':
        columns.remove(feature + '.')
    data.add_columns(data_unpacked[columns])


# In[6]:

features = data.column_names()
features.remove('Party')
features.remove('USER_ID')
features.remove('YOB')  # Remove the response variable
print features
initial_features = features 


# In[7]:

#features = mini_data.column_names()
#features.remove('Party')  # Remove the response variable
mini_data = data[0:3500]
#mini_data


# In[8]:

train_data, validation_set = data.random_split(.8, seed=1)
print len(train_data)
print len(validation_set)
print len(train_data[train_data['Party'] == 'Democrat'])
print len(train_data[train_data['Party'] == 'Republican'])


# In[9]:

def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    
    return len(data) <= min_node_size


# In[10]:

def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    
    gain = error_before_split - error_after_split
    return gain


# In[11]:

def intermediate_node_num_mistakes(labels_in_node):
    #calculates the number of misclassified examples when predicting the majority class.
    
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count of democrat examples
    democrat = sum(labels_in_node == 'Democrat')
    #print democrat
    
    # Count of republican examples
    republican = sum(labels_in_node == 'Republican')        
    #print republican
    
    # Return the number of mistakes that the majority classifier makes.
    mistakes = min(democrat, republican)
    return mistakes


# In[12]:

def best_splitting_feature(data, features, target):
    #finds the best feature to split on given the data and a list of features to consider
    
    best_feature = None     # Keep track of the best feature 
    best_error = 100000     # Keep track of the best error so far 

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1] 
            
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split['Party'])           

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split['Party'])
            
        # Compute the classification error of this split.
        error = float(left_mistakes + right_mistakes) / num_data_points

        if error < best_error:
            best_error = error
            best_feature = feature
    
    return best_feature 


# In[13]:

def create_leaf(target_values):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True }
    
    democrat = len(target_values[target_values == 'Democrat'])
    republican = len(target_values[target_values == 'Republican'])
    
    # For the leaf node, set the prediction to be the majority class.
    if democrat > republican:
        leaf['prediction'] = 'Democrat'       
    else:
        leaf['prediction'] = 'Republican'     
        
    return leaf 


# In[14]:

def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size = 10, 
                         min_error_reduction = 0.002):
    
    remaining_features = features[:] 
    
    target_values = data[target]
    current_error = intermediate_node_num_mistakes(target_values)
    print current_error
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    
    # Stopping condition 1: All nodes are of the same type.
    if  current_error == 0:
        print "Stopping condition 1 reached. All data points have the same target value."                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."                
        return create_leaf(target_values)    
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    if reached_minimum_node_size(data, min_node_size): 
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)  
    
    # Find the best splitting feature.
    splitting_feature = best_splitting_feature(data, features, target)
    print splitting_feature
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Calculate the error before splitting
    error_before_split = current_error / float(len(data))
    
    # Calculate the error after splitting
    left_mistakes = intermediate_node_num_mistakes(left_split[target]) 
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # Early stopping condition 3: Minimum error reduction
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values) 
    
    
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))
    
    # Repeat on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
    


# In[15]:

decision_tree = decision_tree_create(train_data, features, 'Party', max_depth = 15)


# In[16]:

def classify(tree, example):   
    if tree['is_leaf']:
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = example[tree['splitting_feature']]
        
        if split_feature_value == 0:
            return classify(tree['left'], example)
        else:
            return classify(tree['right'], example)


# In[17]:

def evaluate_classification_error(tree, data):
    
    predictions = data.apply(lambda x: classify(tree, x))
    #print predictions
    #print data['Party']
    
    
    #print data['Party'] != predictions
    mistakes = data['Party'] != predictions
    error = mistakes.sum() /  float(len(data))  
    return error


# In[25]:

data.remove_columns(data.column_names())
print data
accuracy_train = evaluate_classification_error(decision_tree, train_data)
print accuracy_train
#accuracy_validation = evaluate_classification_error(decision_tree, validation_set)
#print accuracy_validation


# In[176]:

test_set = graphlab.SFrame.read_csv('test2016.csv/')
test_set.column_names()


# In[177]:

result = graphlab.SFrame.read_csv('sampleSubmission2016.csv/')
result_frame = graphlab.SFrame(result['Predictions'])
result_frame.rename({'X1': 'Party'})
result_frame


# In[178]:

test_set.add_columns(result_frame)
test_set.column_names()


# In[179]:

for feature in initial_features[2:len(initial_features) + 1]:
    test_one_hot_encoded = test_set[feature].apply(lambda x: {x: 1})    
    test_unpacked = test_one_hot_encoded.unpack(column_name_prefix = feature)
    
    # Change None's to 0's
    for column in test_unpacked.column_names():
        test_unpacked[column] = test_unpacked[column].fillna(0)

    
    test_set.remove_column(feature)
    columns = test_unpacked.column_names()
    columns.remove(feature + '.')
    test_set.add_columns(test_unpacked[columns])


# In[180]:

len(test_set.column_names())


# In[181]:

accuracy = evaluate_classification_error(decision_tree, test_set)
print accuracy


# In[ ]:



