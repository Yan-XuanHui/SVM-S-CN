clear;
run_count = 1; %number of test
avg_result = linspace(1,run_count,run_count);
class = 2 ;  %number of class
filePath = '..\data\BreastCancer.xlsx';
C = 10 ;   %parameter C of SVM
i = 1;
while i <= run_count
    fprintf('%dst test:\n',i);
    if(class<=2)
        %result = SVM_Categorical_Fun(filePath,'gaussian', 0.2, C);
        result = SVM_Categorical_Fun(filePath, 'poly', 3, C);
    else
        %result = SVM_Categorical_MultiClass_Fun(filePath, 'gaussian', 0.6, C);
        result = SVM_Categorical_MultiClass_Fun(filePath, 'poly', 3, C);
    end;
    avg_result(i) = result;
    fprintf('WF1=%f\n', result);
    i = i+1;
end
avg_result
fprintf('Avg. WF1_Score=%f\n', mean(avg_result));