%%
% * svm 简单算法设计
%
%% 加载数据
% * 最终data格式：m*n，m样本数，n维度
% * label:m*1  标签必须为-1与1这两类
clc
clear
close all
data = load('data_test2.mat');
data = data.data;
train_data = data(1:end-1,:)';
label = data(end,:)';
[num_data,d] = size(train_data);
data = train_data;
%% 定义向量机参数
alphas = zeros(num_data,1);
% 系数
b = 0;
% 松弛变量影响因子
C = 0.6;
iter = 0;
max_iter = 40;
%%
while iter < max_iter
    alpha_change = 0;
    for i = 1:num_data
        %输出目标值
        pre_Li = (alphas.*label)'*(data*data(i,:)') + b;
        %样本i误差
        Ei = pre_Li - label(i);
        % 满足KKT条件
        if (label(i)*Ei<-0.001 && alphas(i)<C)||(label(i)*Ei>0.001 && alphas(i)>0)
           % 选择一个和 i 不相同的待改变的alpha(2)--alpha(j)
            j = randi(num_data,1);  
            if j == i
                temp = 1;
                while temp
                    j = randi(num_data,1);
                    if j ~= i
                        temp = 0;
                    end
                end
            end
            % 样本j的输出值
            pre_Lj = (alphas.*label)'*(data*data(j,:)') + b;
            %样本j误差
            Ej = pre_Lj - label(j);
            %更新上下限
            if label(i) ~= label(j) %类标签相同
                L = max(0,alphas(j) - alphas(i));
                H = min(C,C + alphas(j) - alphas(i));
            else
                L = max(0,alphas(j) + alphas(i) -C);
                H = min(C,alphas(j) + alphas(i));
            end
            if L==H  %上下限一样结束本次循环
                continue;end
            %计算eta
            eta = 2*data(i,:)*data(j,:)'- data(i,:)*data(i,:)' - ...
                data(j,:)*data(j,:)';
            %保存旧值
            alphasI_old = alphas(i);
            alphasJ_old = alphas(j);
            %更新alpha(2)，也就是alpha(j)
            alphas(j) = alphas(j) - label(j)*(Ei-Ej)/eta;
            %限制范围
            if alphas(j) > H
                alphas(j) = H;
            elseif alphas(j) < L
                    alphas(j) = L;
            end
            %如果alpha(j)没怎么改变，结束本次循环
            if abs(alphas(j) - alphasJ_old)<1e-4
                continue;end
            %更新alpha(1)，也就是alpha(i)
            alphas(i) = alphas(i) + label(i)*label(j)*(alphasJ_old-alphas(j));
            %更新系数b
            b1 = b - Ei - label(i)*(alphas(i)-alphasI_old)*data(i,:)*data(i,:)'-...
                label(j)*(alphas(j)-alphasJ_old)*data(i,:)*data(j,:)';
            b2 = b - Ej - label(i)*(alphas(i)-alphasI_old)*data(i,:)*data(j,:)'-...
                label(j)*(alphas(j)-alphasJ_old)*data(j,:)*data(j,:)';
            %b的几种选择机制
            if alphas(i)>0 && alphas(i)<C
                b = b1;
            elseif alphas(j)>0 && alphas(j)<C
                b = b2;
            else
                b = (b1+b2)/2;
            end
            %确定更新了，记录一次
            alpha_change = alpha_change + 1;
        end
    end
    % 没有实行alpha交换，迭代加1
    if alpha_change == 0
        iter = iter + 1;
    %实行了交换，迭代清0
    else
        iter = 0;
    end
    disp(['iter ================== ',num2str(iter)]);
end
%% 计算权值W
W = (alphas.*label)'*data;
%记录支持向量位置
index_sup = find(alphas ~= 0);
%计算预测结果
predict = (alphas.*label)'*(data*data') + b;
predict = sign(predict);
%% 显示结果
figure;
index1 = find(predict==-1);
data1 = (data(index1,:))';
plot(data1(1,:),data1(2,:),'+r');
hold on
index2 = find(predict==1);
data2 = (data(index2,:))';
plot(data2(1,:),data2(2,:),'*');
hold on
dataw = (data(index_sup,:))';
plot(dataw(1,:),dataw(2,:),'og','LineWidth',2);
% 画出分界面，以及b上下正负1的分界面
hold on
k = -W(1)/W(2);
x = 1:0.1:5;
y = k*x + b;
plot(x,y,x,y-1,'r--',x,y+1,'r--');
title(['松弛变量范围C = ',num2str(C)]);