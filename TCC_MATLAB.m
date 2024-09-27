clear

%Carrega dataset
dataset = load('dataset.csv');
load('cvp.mat');

% %Adicionar Nível e Temperatura no csv
% dataset = load('Vazao_L100-T29.csv');
% for i=1:length(dataset)
%     A(i)=100;
%     B(i)=29;
% end
% A=cat(2,A',B');
% dataset=cat(2,dataset,A);
% writematrix(dataset,'Vazao_L100-T29-Modificado.csv');

% %Unir datasets
% dataset1 = load('Vazao_L50-T29-Modificado.csv');
% dataset2 = load('Vazao_L50-T65-Modificado.csv');
% dataset3 = load('Vazao_L100-T29-Modificado.csv');
% dataset4 = load('Vazao_L100-T65-Modificado.csv');
% A=cat(1,dataset1,dataset2);
% B=cat(1,dataset3,dataset4);
% dataset=cat(1,A,B);
% writematrix(dataset,'dataset.csv')

input = dataset(:,[2 6 7 9]);
output = dataset(:,1);

% input_2 = dataset(:,[2 7]);
% output_2 = dataset(:,1);
% 
% j=1;
% for i=1:1000:length(input_2)
% 
%     input(j,:)=input_2(i,:);
%     output(j,:)=output_2(i,:);
%     j=j+1;
% end

%Verifica correlacao entre variáveis
correlacao=corr(dataset);

%Treina o modelo
tic
%model_lm = stepwiselm(input,output)
model_lm = fitlm(input,output)
Time=toc;

%Testa o modelo
testClass = predict(model_lm,input);

%plot(testClass,output);

RMSE = sqrt(mean((output(:,1) - testClass(:,1)).^2))

% figure(1)
% plot(dataset(:,[2]),output,'ro')
% title('Speed / Flow Rate')
% xlabel('Speed')
% ylabel('Flow Rate')
% 
% figure(2)
% plot(dataset(:,[3]),output,'ro')
% title('Current / Flow Rate')
% xlabel('Current')
% ylabel('Flow Rate')
% 
% figure(3)
% plot(dataset(:,[5]),output,'ro')
% title('Power Factor / Flow Rate')
% xlabel('Power Factor')
% ylabel('Flow Rate')
% 
% figure(4)
% plot(dataset(:,[6]),output,'ro')
% title('Output Voltage / Flow Rate')
% xlabel('Output Voltage')
% ylabel('Flow Rate')
% 
% figure(5)
% plot(dataset(:,[7]),output,'ro')
% title('Actual Voltage / Flow Rate')
% xlabel('Actual Voltage')
% ylabel('Flow Rate')
