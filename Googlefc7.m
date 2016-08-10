% [scores, maxlabel] = classification_demo(im, use_gpu)
%
% Image classification demo using BVLC CaffeNet.
%
% IMPORTANT: before you run this demo, you should download BVLC CaffeNet
% from Model Zoo (http://caffe.berkeleyvision.org/model_zoo.html)
%
% ****************************************************************************
% For detailed documentation and usage on Caffe's Matlab interface, please
% refer to Caffe Interface Tutorial at
% http://caffe.berkeleyvision.org/tutorial/interfaces.html#matlab
% ****************************************************************************
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%   maxlabel the label of the highest score
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = classification_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format in W x H x C with BGR channels:
%   % permute channels from RGB to BGR
%   im_data = im(:, :, [3, 2, 1]);
%   % flip width and height to make width the fastest dimension
%   im_data = permute(im_data, [2, 1, 3]);
%   % convert from uint8 to single
%   im_data = single(im_data);
%   % reshape to a fixed size (e.g., 227x227).
%   im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % subtract mean_data (already in W x H x C with BGR channels)
%   im_data = im_data - mean_data;

% If you have multiple images, cat them with cat(4, ...)

% Add caffe/matlab to you Matlab search PATH to use matcaffe


  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
% else
%   caffe.set_mode_cpu();
% end


% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
% model_dir = '../../models/bvlc_reference_caffenet/';
% net_model = 'BN_deploy.prototxt';
% net_weights = 'caffenet_BN_train_iter_73075.caffemodel';

net_model = '/home/pipag/下载/fxq/demo/GoogleCar/deploy.prototxt';
net_weights = '/home/pipag/下载/fxq/demo/GoogleCar/googlenet_finetune_web_car_iter_10000.caffemodel';

phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

tt2=importdata('val_list.txt');
names=importdata('val_names.txt');
[m2,n2]=size(tt2);
feature_test_set=zeros(m2,431);



% prepare oversampled input
% input_data is Height x Width x Channel x Num

%input_data = {prepare_image(im)};
for ss =1:m2
im=imread(['/home/pipag/下载/fxq/demo/val/images/',tt2{ss,1}]);
input_data={prepare_image(im)};
net.forward(input_data);
feature=net.blobs('loss3_classifier_model').get_data();

xxx=mean(feature,2);
feature_test_set(ss,:)=xxx';
fprintf('%d %s\n',ss,'th image extracted');% ��ʾ���ڴ����·����ͼ����
end
save Google_loss3_classifier_model.mat feature_test_set;

%reduce dimension by PCA



%L2 normalize
featurenorm=normalize1(feature_test_set);
save Google_loss3_classifier_model_norm.mat featurenorm;


% cal caffe.reset_all() to reset caffe
caffe.reset_all();

% ------------------------------------------------------------------------

