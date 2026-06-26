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

    tokens = regexp(filename,'sub_(\d+)','tokens');
    if isempty(tokens)
        continue;
    end

    subject = str2double(tokens{1}{1});

    fprintf('Processing %s\n', filename);

    data = load(fullfile(folder,filename));
    eeg = data.Clean_data;

    epoch_len = 4 * fs;
    num_epochs = floor(size(eeg,2)/epoch_len);

    % LABELS
    if contains(lower(filename),'arithmetic')
        label = 0;
    elseif contains(lower(filename),'mirror')
        label = 1;
    elseif contains(lower(filename),'stroop')
        label = 2;
    else
        continue;
    end

    for k = 1:num_epochs

        idx1 = (k-1)*epoch_len + 1;
        idx2 = k*epoch_len;

        epoch = eeg(:,idx1:idx2);

        feat = extract_features(epoch,fs);   % YOUR FUNCTION

        X = [X; feat];
        y = [y; label];
        subjectIDs = [subjectIDs; subject];

    end
end

%% =========================
% VERIFY DATASET
%% =========================
fprintf('\n===== DATASET INFO =====\n');
fprintf('Features: %d x %d\n', size(X,1), size(X,2));

assert(size(X,1) == length(y), 'Mismatch X and y');

disp('Class distribution:');
tabulate(y)

disp('Subjects:');
disp(unique(subjectIDs)');

%% =========================
% 10-FOLD CV
%% =========================
fprintf('\n===== 10-FOLD CV =====\n');

cv10 = cvpartition(y,'KFold',10);

svmFold   = zeros(cv10.NumTestSets,1);
rfFold    = zeros(cv10.NumTestSets,1);
bagFold   = zeros(cv10.NumTestSets,1);
adaFold   = zeros(cv10.NumTestSets,1);
knnFold   = zeros(cv10.NumTestSets,1);
treeFold  = zeros(cv10.NumTestSets,1);
nbFold    = zeros(cv10.NumTestSets,1);

for fold = 1:cv10.NumTestSets

    trainIdx = training(cv10,fold);
    testIdx  = test(cv10,fold);

    Xtrain = X(trainIdx,:);
    Ytrain = y(trainIdx);

    Xtest  = X(testIdx,:);
    Ytest  = y(testIdx);

    %% FEATURE SELECTION
    [idx,~] = fscmrmr(Xtrain,Ytrain);
    selK = min(30,size(Xtrain,2));

    Xtrain = Xtrain(:,idx(1:selK));
    Xtest  = Xtest(:,idx(1:selK));

    %% NORMALIZATION
    mu = mean(Xtrain);
    sigma = std(Xtrain) + 1e-8;

    Xtrain = (Xtrain - mu) ./ sigma;
    Xtest  = (Xtest - mu) ./ sigma;

    %% =========================
    % 1. SVM (RBF)
    %% =========================
    t = templateSVM('KernelFunction','rbf','Standardize',false);
    svmModel = fitcecoc(Xtrain,Ytrain,'Learners',t);
    pred = predict(svmModel,Xtest);
    svmFold(fold) = mean(pred==Ytest)*100;

    %% =========================
    % 2. RANDOM FOREST
    %% =========================
    rfModel = TreeBagger(150,Xtrain,Ytrain,'Method','classification');
    pred = str2double(predict(rfModel,Xtest));
    rfFold(fold) = mean(pred==Ytest)*100;

    %% =========================
    % 3. BAGGING (VERY STABLE EEG MODEL)
    %% =========================
    bagModel = fitcensemble(Xtrain,Ytrain, ...
        'Method','Bag', ...
        'NumLearningCycles',200);

    pred = predict(bagModel,Xtest);
    bagFold(fold) = mean(pred==Ytest)*100;

    %% =========================
    % 4. ADABOOSTM2 (MULTICLASS BOOSTING)
    %% =========================
    adaModel = fitcensemble(Xtrain,Ytrain, ...
        'Method','AdaBoostM2', ...
        'NumLearningCycles',300);

    pred = predict(adaModel,Xtest);
    adaFold(fold) = mean(pred==Ytest)*100;

    %% =========================
    % 5. KNN
    %% =========================
    knnModel = fitcknn(Xtrain,Ytrain,'NumNeighbors',5);
    pred = predict(knnModel,Xtest);
    knnFold(fold) = mean(pred==Ytest)*100;

    %% =========================
    % 6. DECISION TREE
    %% =========================
    treeModel = fitctree(Xtrain,Ytrain);
    pred = predict(treeModel,Xtest);
    treeFold(fold) = mean(pred==Ytest)*100;

    %% =========================
    % 7. NAIVE BAYES
    %% =========================
    nbModel = fitcnb(Xtrain,Ytrain);
    pred = predict(nbModel,Xtest);
    nbFold(fold) = mean(pred==Ytest)*100;

end

%% =========================
% RESULTS 10-FOLD
%% =========================
fprintf('\n===== 10-FOLD RESULTS =====\n');

fprintf('SVM        : %.2f ± %.2f\n', mean(svmFold), std(svmFold));
fprintf('RF         : %.2f ± %.2f\n', mean(rfFold), std(rfFold));
fprintf('Bagging    : %.2f ± %.2f\n', mean(bagFold), std(bagFold));
fprintf('AdaBoost   : %.2f ± %.2f\n', mean(adaFold), std(adaFold));
fprintf('KNN        : %.2f ± %.2f\n', mean(knnFold), std(knnFold));
fprintf('Tree       : %.2f ± %.2f\n', mean(treeFold), std(treeFold));
fprintf('NaiveBayes : %.2f ± %.2f\n', mean(nbFold), std(nbFold));

%% =========================
% LOSO VALIDATION (BEST MODELS)
%% =========================
fprintf('\n===== LOSO =====\n');

subjects = unique(subjectIDs);

svmLOSO = zeros(length(subjects),1);
bagLOSO = zeros(length(subjects),1);
adaLOSO = zeros(length(subjects),1);

for s = 1:length(subjects)

    testSubject = subjects(s);

    trainIdx = subjectIDs ~= testSubject;
    testIdx  = subjectIDs == testSubject;

    Xtrain = X(trainIdx,:);
    Ytrain = y(trainIdx);

    Xtest  = X(testIdx,:);
    Ytest  = y(testIdx);

    %% FEATURE SELECTION
    [idx,~] = fscmrmr(Xtrain,Ytrain);
    selK = min(30,size(Xtrain,2));

    Xtrain = Xtrain(:,idx(1:selK));
    Xtest  = Xtest(:,idx(1:selK));

    %% NORMALIZATION
    mu = mean(Xtrain);
    sigma = std(Xtrain) + 1e-8;

    Xtrain = (Xtrain - mu) ./ sigma;
    Xtest  = (Xtest - mu) ./ sigma;

    %% SVM
    t = templateSVM('KernelFunction','rbf','Standardize',false);
    svmModel = fitcecoc(Xtrain,Ytrain,'Learners',t);
    svmLOSO(s) = mean(predict(svmModel,Xtest)==Ytest)*100;

    %% BAGGING
    bagModel = fitcensemble(Xtrain,Ytrain,'Method','Bag','NumLearningCycles',200);
    bagLOSO(s) = mean(predict(bagModel,Xtest)==Ytest)*100;

    %% ADABOOST
    adaModel = fitcensemble(Xtrain,Ytrain,'Method','AdaBoostM2','NumLearningCycles',300);
    adaLOSO(s) = mean(predict(adaModel,Xtest)==Ytest)*100;

    fprintf('Subject %d done\n', testSubject);

end

%% =========================
% FINAL LOSO RESULTS
%% =========================
fprintf('\n===== FINAL LOSO RESULTS =====\n');

fprintf('SVM      : %.2f ± %.2f\n', mean(svmLOSO), std(svmLOSO));
fprintf('Bagging  : %.2f ± %.2f\n', mean(bagLOSO), std(bagLOSO));
fprintf('AdaBoost : %.2f ± %.2f\n', mean(adaLOSO), std(adaLOSO));