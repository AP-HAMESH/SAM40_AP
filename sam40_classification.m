clc;
clear;
close all;

%% SETTINGS

fs = 128;

folder = 'C:\Users\AP HAMESH\OneDrive\Desktop\AP\MIND\sam40\Data\SAM40\filtered_data';

files = dir(fullfile(folder,'*.mat'));
X = [];
y = [];
subjectIDs = [];

%% FEATURE EXTRACTION

for i = 1:length(files)

    filename = files(i).name;
    %changed
    tokens = regexp(filename,'sub_(\d+)','tokens');

    if isempty(tokens)
        continue;
    end
    subject = str2double(tokens{1}{1});
     %changed

    fprintf('Processing %s\n', filename);

    data = load(fullfile(folder,filename));

    eeg = data.Clean_data;

    epoch_len = 4 * fs;

    num_epochs = floor(size(eeg,2)/epoch_len);

    % Assign label once per file
    if contains(lower(filename),'arithmetic')

        label = 0;

    elseif contains(lower(filename),'mirror')

        label = 1;

    elseif contains(lower(filename),'stroop')

        label = 2;

    else

        fprintf('Skipping file: %s\n', filename);
        continue;

    end

    for k = 1:num_epochs

        idx1 = (k-1)*epoch_len + 1;
        idx2 = k*epoch_len;

        epoch = eeg(:,idx1:idx2);

        feat = extract_features(epoch,fs);

        X = [X; feat];

        y = [y; label];

        subjectIDs = [subjectIDs; subject];

    end
end

%% VERIFY DATASET

fprintf('\n===== DATASET INFO =====\n');

fprintf('Feature Rows = %d\n',size(X,1));
fprintf('Feature Cols = %d\n',size(X,2));

fprintf('Label Rows   = %d\n',length(y));

assert(size(X,1)==length(y),...
       'Feature/Label mismatch!');

disp('Unique Classes:')
disp(unique(y))

disp('Class Counts:')
tabulate(y)
disp('Subjects found:')
disp(unique(subjectIDs)')
%% NORMALIZATION

X = zscore(X);

%K FOLD CROSS VALIDATION
fprintf('\n===== 10-FOLD CROSS VALIDATION =====\n');

cv10 = cvpartition(y,'KFold',10);

svmFold = zeros(cv10.NumTestSets,1);
rfFold  = zeros(cv10.NumTestSets,1);
boostFold = zeros(cv10.NumTestSets,1);

for fold = 1:cv10.NumTestSets

    trainIdx = training(cv10,fold);
    testIdx  = test(cv10,fold);

    Xtrain = X(trainIdx,:);
    Ytrain = y(trainIdx);

    Xtest = X(testIdx,:);
    Ytest = y(testIdx);

    % SVM
    svmModel = fitcecoc(Xtrain,Ytrain);
    pred = predict(svmModel,Xtest);
    svmFold(fold) = mean(pred==Ytest)*100;

    % Random Forest
    rfModel = TreeBagger(100,Xtrain,Ytrain,...
        'Method','classification');

    pred = str2double(predict(rfModel,Xtest));
    rfFold(fold) = mean(pred==Ytest)*100;

  % Bagged Trees (stable multiclass ensemble)

boostModel = fitcensemble(...
    Xtrain,...
    Ytrain,...
    'Method','Bag',...
    'NumLearningCycles',200);

pred = predict(boostModel,Xtest);

boostFold(fold) = mean(pred==Ytest)*100;

end

fprintf('10-Fold SVM        : %.2f +/- %.2f\n', ...
    mean(svmFold),std(svmFold));

fprintf('10-Fold RF         : %.2f +/- %.2f\n', ...
    mean(rfFold),std(rfFold));

fprintf('10-Fold Boost      : %.2f +/- %.2f\n', ...
    mean(boostFold),std(boostFold));


%LOSO

fprintf('\n===== LOSO VALIDATION =====\n');

subjects = unique(subjectIDs);

svmLOSO = zeros(length(subjects),1);
rfLOSO = zeros(length(subjects),1);
boostLOSO = zeros(length(subjects),1);

for s = 1:length(subjects)

    testSubject = subjects(s);

    trainIdx = subjectIDs ~= testSubject;
    testIdx  = subjectIDs == testSubject;

    Xtrain = X(trainIdx,:);
    Ytrain = y(trainIdx);

    Xtest = X(testIdx,:);
    Ytest = y(testIdx);

    % SVM
    svmModel = fitcecoc(Xtrain,Ytrain);

    pred = predict(svmModel,Xtest);

    svmLOSO(s) = mean(pred==Ytest)*100;

    % Random Forest
    rfModel = TreeBagger(100,...
        Xtrain,...
        Ytrain,...
        'Method','classification');

    pred = str2double(predict(rfModel,Xtest));

    rfLOSO(s) = mean(pred==Ytest)*100;

    % Boosted Trees
    boostModel = fitcensemble(...
    Xtrain,...
    Ytrain,...
    'Method','Bag',...
    'NumLearningCycles',200);
    pred = predict(boostModel,Xtest);

    boostLOSO(s) = mean(pred==Ytest)*100;

    fprintf('Subject %d Done\n',testSubject);

end

fprintf('\n===== LOSO RESULTS =====\n');

fprintf('LOSO SVM        : %.2f +/- %.2f\n', ...
    mean(svmLOSO),std(svmLOSO));

fprintf('LOSO RF         : %.2f +/- %.2f\n', ...
    mean(rfLOSO),std(rfLOSO));

fprintf('LOSO Boost      : %.2f +/- %.2f\n', ...
    mean(boostLOSO),std(boostLOSO));

%% TRAIN TEST SPLIT

cv = cvpartition(y,'HoldOut',0.2);

Xtrain = X(training(cv),:);
Ytrain = y(training(cv));

Xtest = X(test(cv),:);
Ytest = y(test(cv));

%% SVM

svmModel = fitcecoc(Xtrain,Ytrain);

svmPred = predict(svmModel,Xtest);

svmAcc = mean(svmPred==Ytest)*100;

%% RANDOM FOREST

rfModel = TreeBagger(100,...
                     Xtrain,...
                     Ytrain,...
                     'Method','classification');

rfPred = predict(rfModel,Xtest);

rfPred = str2double(rfPred);

rfAcc = mean(rfPred==Ytest)*100;

%% KNN

knnModel = fitcknn(Xtrain,...
                   Ytrain,...
                   'NumNeighbors',5);

knnPred = predict(knnModel,Xtest);

knnAcc = mean(knnPred==Ytest)*100;

%% DECISION TREE

treeModel = fitctree(Xtrain,Ytrain);

treePred = predict(treeModel,Xtest);

treeAcc = mean(treePred==Ytest)*100;

%% NAIVE BAYES

nbModel = fitcnb(Xtrain,Ytrain);

nbPred = predict(nbModel,Xtest);

nbAcc = mean(nbPred==Ytest)*100;

%% BOOSTED TREES (XGBoost-like)

boostModel = fitcensemble(...
    Xtrain,...
    Ytrain,...
    'Method','Bag',...
    'NumLearningCycles',200);
boostPred = predict(boostModel,Xtest);

boostAcc = mean(boostPred==Ytest)*100;

%% CONFUSION MATRIX FOR BEST MODEL

figure;
confusionchart(Ytest,boostPred);
title('Boosted Trees Confusion Matrix');

%% RESULTS

fprintf('\n===============================\n');
fprintf('SAM40 CLASSIFICATION RESULTS\n');
fprintf('===============================\n');

fprintf('SVM Accuracy          : %.2f%%\n',svmAcc);
fprintf('Random Forest         : %.2f%%\n',rfAcc);
fprintf('KNN Accuracy          : %.2f%%\n',knnAcc);
fprintf('Decision Tree         : %.2f%%\n',treeAcc);
fprintf('Naive Bayes           : %.2f%%\n',nbAcc);
fprintf('Boosted Trees         : %.2f%%\n',boostAcc);