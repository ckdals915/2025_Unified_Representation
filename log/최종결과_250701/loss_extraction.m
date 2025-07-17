
% file_standard = 'SEED0_MLPMNIST_bs128.txt';
% file_unified    = 'SEED0_MLPAsGNNMNIST_bs128.txt';
% file_standard = 'SEED3_CNN1DCWRU_bs128_CWRU_lr5e-05.txt';
% file_unified    = 'SEED3_CNN1DCWRUAsGNN_bs128_CWRU_lr5e-05.txt';
file_standard = 'SEED5_ResNet18_bs128.txt';
file_unified    = 'SEED5_ResNet18AsGNN_bs128.txt';
% file_standard = 'SEED2_VGG_bs16.txt';
% file_unified    = 'SEED2_VGGAsGNN_bs16.txt';
% file_standard = 'SEED3_ViT_bs128_CIFAR100_lr0.0001.txt';
% file_unified    = 'SEED3_ViTAsGNN_bs128_CIFAR100_lr0.0001.txt';




[train_std, val_std, train_acc_std, val_acc_std] = readLossesFromLog(file_standard);
[train_asg, val_asg, train_acc_asg, val_acc_asg]  = readLossesFromLog(file_unified);

train_acc_std = train_acc_std .* 100;
val_acc_std = val_acc_std .* 100;
train_acc_asg = train_acc_asg .* 100;
val_acc_asg = val_acc_asg .* 100;

% Loss
figure('Position', [100 100 800 600]);
hold on;

baseColor = [0 0.4470 0.7410];


plot(1:numel(train_std), train_std, ...
    'Color', [baseColor 0.3], 'LineWidth', 4);
plot(1:numel(val_std), val_std, ...
    '--', 'Color', [baseColor 0.3], 'LineWidth', 4);

plot(1:numel(train_asg), train_asg, ...
    'Color', [baseColor 1], 'LineWidth', 4);
plot(1:numel(val_asg), val_asg, ...
    '--', 'Color', [baseColor 1], 'LineWidth', 4);

ax = gca;
ax.FontSize   = 20;
ax.FontWeight = 'bold';


hlegend= legend({ ...
    'Reference Train', ...
    'Reference Val', ...
    'Unified Train', ...
    'Unified Val'}, ...
    'Location', 'northeast');
hLegend.FontSize   = 20;
hLegend.FontWeight = 'bold';

xlabel('Epoch', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Loss',  'FontSize', 20, 'FontWeight', 'bold');

title('Train & Validation Loss', ...
    'FontSize', 22, 'FontWeight', 'bold');

grid on;
hold off;


% ACC
figure('Position', [100 100 800 600]);
hold on;

baseColor = [0 0.4470 0.7410];


plot(1:numel(train_acc_std), train_acc_std, ...
    'Color', [baseColor 0.3], 'LineWidth', 4);
plot(1:numel(val_acc_std), val_acc_std, ...
    '--', 'Color', [baseColor 0.3], 'LineWidth', 4);

plot(1:numel(train_acc_asg), train_acc_asg, ...
    'Color', [baseColor 1], 'LineWidth', 4);
plot(1:numel(val_acc_asg), val_acc_asg, ...
    '--', 'Color', [baseColor 1], 'LineWidth', 4);

ax = gca;
ax.FontSize   = 20;
ax.FontWeight = 'bold';


hlegend= legend({ ...
    'Reference Train', ...
    'Reference Val', ...
    'Unified Train', ...
    'Unified Val'}, ...
    'Location', 'east');
hLegend.FontSize   = 20;
hLegend.FontWeight = 'bold';

xlabel('Epoch', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Accuraty(%)',  'FontSize', 20, 'FontWeight', 'bold');

title('Train & Validation Accuracy', ...
    'FontSize', 22, 'FontWeight', 'bold');

grid on;
hold off;


%%
clear; close all; clc;

% ------------------------------------------------------------------------
% 1) 모델별 파일명 패턴 및 SEED 범위 정의
% ------------------------------------------------------------------------
%   - name: 모델 이름
%   - stdPattern: Reference(standard) 로그 파일명 패턴
%   - unfPattern: Unified 로그 파일명 패턴
%   - seedRange: 이 모델에 대해 사용할 SEED 범위 (예: 0:9 또는 0:2)
%
models = { ...
    struct('name','MLP',    ...
           'stdPattern','SEED%d_MLPMNIST_bs128.txt',          ...
           'unfPattern','SEED%d_MLPAsGNNMNIST_bs128.txt',     ...
           'seedRange',0:9), ...
    struct('name','CNN1D',  ...
           'stdPattern','SEED%d_CNN1DCWRU_bs128_CWRU_lr5e-05.txt', ...
           'unfPattern','SEED%d_CNN1DCWRUAsGNN_bs128_CWRU_lr5e-05.txt', ...
           'seedRange',[1,2,3,4,5,7]), ...
    struct('name','ResNet', ...
           'stdPattern','SEED%d_ResNet18_bs128.txt',         ...
           'unfPattern','SEED%d_ResNet18AsGNN_bs128.txt',   ...
           'seedRange',0:9), ...
    struct('name','VGG',    ...
           'stdPattern','SEED%d_VGG_bs16.txt',              ...
           'unfPattern','SEED%d_VGGAsGNN_bs16.txt',        ...
           'seedRange',0:2), ...  % <-- VGG만 SEED 0~2
    struct('name','ViT',    ...
           'stdPattern','SEED%d_ViT_bs128_CIFAR100_lr0.0001.txt',       ...
           'unfPattern','SEED%d_ViTAsGNN_bs128_CIFAR100_lr0.0001.txt',  ...
           'seedRange',0:9) ...
};

numModels = numel(models);

% 결과를 저장할 구조체 초기화
results = struct();
for m = 1:numModels
    results(m).name    = models{m}.name;
    % seedRange 길이에 맞춰 빈 벡터 할당 (초기엔 NaN)
    sr = models{m}.seedRange;
    results(m).seeds   = sr;
    results(m).bestStd = nan(1, numel(sr));
    results(m).bestUnf = nan(1, numel(sr));
end

% ------------------------------------------------------------------------
% 2) 모델별 반복: 각 SEED마다 최고 Validation Accuracy 추출
% ------------------------------------------------------------------------
for m = 1:numModels
    modelName  = models{m}.name;
    stdPattern = models{m}.stdPattern;
    unfPattern = models{m}.unfPattern;
    seedList   = models{m}.seedRange;
    numSeeds   = numel(seedList);
    
    fprintf('=== Processing %s (SEEDs: %s) ===\n', modelName, mat2str(seedList));
    
    for idx = 1:numSeeds
        seed = seedList(idx);
        
        % 실제 파일명 생성
        file_standard = sprintf(stdPattern, seed);
        file_unified  = sprintf(unfPattern, seed);
        
        % 로그 읽어오기 (readLossesFromLog 함수가 반드시 현재 경로에 있어야 함)
        try
            [~, ~, ~, valAcc_std] = readLossesFromLog(file_standard);
            [~, ~, ~, valAcc_unf] = readLossesFromLog(file_unified);
        catch ME
            error('파일을 열 수 없습니다: %s 또는 %s\n오류 메시지: %s', file_standard, file_unified, ME.message);
        end
        
        % 0~1 스케일 → 0~100(%) 변환
        valAcc_std = valAcc_std * 100;
        valAcc_unf = valAcc_unf * 100;
        
        if strcmp(modelName, 'CNN1D')
            maxEpoch = 200;
            % valAcc_* 의 길이가 200보다 짧으면 end까지, 아니면 200까지 자름
            cutoff = min(maxEpoch, numel(valAcc_std));
            valAcc_std = valAcc_std(1:cutoff);
            
            cutoff = min(maxEpoch, numel(valAcc_unf));
            valAcc_unf = valAcc_unf(1:cutoff);
        end

        % 각 SEED별 최고 Validation Accuracy 추출
        results(m).bestStd(idx) = max(valAcc_std);
        results(m).bestUnf(idx) = max(valAcc_unf);
        
        fprintf('  SEED %d: Std = %.2f%%, Unf = %.2f%%\n', ...
                seed, results(m).bestStd(idx), results(m).bestUnf(idx));
    end
    fprintf('\n');
end

% ------------------------------------------------------------------------
% 3) 모델별 평균 및 표준편차 계산 (양쪽 극단값 제거)
% ------------------------------------------------------------------------
for m = 1:numModels
    modelName = results(m).name;
    valuesStd = results(m).bestStd;   % 이 모델의 SEED별 Reference 최고값
    valuesUnf = results(m).bestUnf;   % 이 모델의 SEED별 Unified  최고값
    
    % (1) Reference(Std) 모델: 정렬 → 2:end-1 → mean, std
    sortedStd  = sort(valuesStd);
    if numel(sortedStd) > 3
        trimmedStd = sortedStd(3:end-2);
    else
        % SEED 갯수가 2 이하라면 ‘양쪽 outlier 제거’ 후 남는 값이 없으므로
        % 그냥 그대로 사용하고 std는 0으로 처리
        trimmedStd = sortedStd;
    end
    meanStd = mean(trimmedStd);
    stdStd  = std(trimmedStd);
    
    % (2) Unified 모델: 정렬 → 2:end-1 → mean, std
    sortedUnf  = sort(valuesUnf);
    if numel(sortedUnf) > 3
        trimmedUnf = sortedUnf(3:end-2);
    else
        trimmedUnf = sortedUnf;
    end
    meanUnf = mean(trimmedUnf);
    stdUnf  = std(trimmedUnf);
    
    % 결과 저장
    results(m).meanStd = meanStd;
    results(m).stdStd  = stdStd;
    results(m).meanUnf = meanUnf;
    results(m).stdUnf  = stdUnf;
    
    % 화면 출력
    fprintf('>>> %s 모델 최종 결과 (양쪽 극단 제거) <<<\n', modelName);
    fprintf('   Reference: 평균 = %.2f%%, 표준편차 = %.2f%%  (SEED 개수: %d → trimmed %d)\n', ...
            meanStd, stdStd, numel(valuesStd), numel(trimmedStd));
    fprintf('   Unified:   평균 = %.2f%%, 표준편차 = %.2f%%  (SEED 개수: %d → trimmed %d)\n\n', ...
            meanUnf, stdUnf, numel(valuesUnf), numel(trimmedUnf));
end

% ------------------------------------------------------------------------
% 4) (선택) 모델별 요약 테이블 생성
% ------------------------------------------------------------------------
% 모델 이름, SEED 개수, trimmed 개수, 평균/표준편차를 한눈에 보기
Summary = cell(numModels,6);
for m = 1:numModels
    Summary{m,1} = results(m).name;
    Summary{m,2} = numel(results(m).seeds);
    % trimmed 개수는 seeds 갯수에서 극단 2개 제거하거나, SEED 개수 ≤ 2면 그대로
    if numel(results(m).seeds) > 2
        Summary{m,3} = numel(results(m).seeds) - 2;
    else
        Summary{m,3} = numel(results(m).seeds);
    end
    Summary{m,4} = results(m).meanStd;
    Summary{m,5} = results(m).stdStd;
    Summary{m,6} = results(m).meanUnf;
    % stdUnf 출력 안 해도 되지만, 일관성을 위해 추가할 수도 있음
    Summary{m,7} = results(m).stdUnf;
end

T = cell2table(Summary, ...
    'VariableNames', {'Model','Total_SEEDs','Trimmed_count','Mean_Ref','Std_Ref','Mean_Unf', 'Std_Unf'} );
disp('=== 모델별 요약 테이블 ===');
disp(T);

function [trainLoss, valLoss, trainAcc, valAcc] = readLossesFromLog(filename)
    fid = fopen(filename, 'r');
    if fid < 0
        error('파일을 열 수 없습니다: %s', filename);
    end

    trainLoss = [];
    valLoss   = [];
    trainAcc = [];
    valAcc = [];

    tline = fgetl(fid);
    while ischar(tline)
        % % “Train Loss: <숫자>” 패턴 추출
        % exprTrain = 'Train\s+Loss:\s*([0-9]*\.?[0-9]+)';
        % tokTrain = regexp(tline, exprTrain, 'tokens');
        % if ~isempty(tokTrain)
        %     trainLoss(end+1) = str2double(tokTrain{1}{1}); %#ok<AGROW>
        % end
        % 
        % % “Val   Loss: <숫자>” 패턴 추출
        % exprVal = 'Val\s+Loss:\s*([0-9]*\.?[0-9]+)';
        % tokVal = regexp(tline, exprVal, 'tokens');
        % if ~isempty(tokVal)
        %     valLoss(end+1) = str2double(tokVal{1}{1}); %#ok<AGROW>
        % end
        % “Train Loss: <숫자>, Train Acc: <숫자>” 패턴 추출
        exprTrain = 'Train\s+Loss:\s*([0-9]*\.?[0-9]+),\s*Train\s+Acc:\s*([0-9]*\.?[0-9]+)';
        tokTrain = regexp(tline, exprTrain, 'tokens');
        if ~isempty(tokTrain)
            trainLoss(end+1) = str2double(tokTrain{1}{1}); %#ok<AGROW>
            trainAcc(end+1)  = str2double(tokTrain{1}{2}); %#ok<AGROW>
        end

        % “Val   Loss: <숫자>, Val   Acc: <숫자>” 패턴 추출
        exprVal = 'Val\s+Loss:\s*([0-9]*\.?[0-9]+),\s*Val\s+Acc:\s*([0-9]*\.?[0-9]+)';
        tokVal = regexp(tline, exprVal, 'tokens');
        if ~isempty(tokVal)
            valLoss(end+1) = str2double(tokVal{1}{1}); %#ok<AGROW>
            valAcc(end+1)  = str2double(tokVal{1}{2}); %#ok<AGROW>
        end

        tline = fgetl(fid);
    end

    fclose(fid);
end
