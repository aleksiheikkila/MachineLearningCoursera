function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

% Try different epsilon values
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % In this part of the exercise, you will implement an algorithm to select
    % the threshold " using the F1 score on a cross validation set.
    % CV-setistä siis tiedetään mitkä olivat anomalioita, l. ~supervised
    
 
% 1 if pval is less than the current epsilon 
cvPredictions = pval < epsilon;
   
% Count the confusion matrix values 
fp = sum( (cvPredictions == 1) & (yval == 0) );
tp = sum( (cvPredictions == 1) & (yval == 1) );
fn = sum( (cvPredictions == 0) & (yval == 1) );
tn = sum( (cvPredictions == 0) & (yval == 0) );

% Precision
prec = tp / (tp+fp);

% Recall
rec = tp / (tp+fn);

% F1 score
F1 = (2*prec*rec) / (prec + rec);





    % =============================================================

      % So far the best epsilon...
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
% End epsilon loop:
end

% End function
end
