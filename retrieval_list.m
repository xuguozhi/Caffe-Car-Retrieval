load CaffeNet_fc7_norm.mat featurenorm;

query_list=importdata('query_list.txt');
ref_list=importdata('ref_list.txt');

fid=fopen('results_CaffeNet_fc7_norm.txt','w')
[n,d]=size(query_list);
[refn,refd]=size(ref_list);
for q=1:n
zz=featurenorm(refn+q,:);
score=zeros(refn,1);
for loop=1:refn
    Vectemp=featurenorm(loop,:);
    score(loop)=zz*Vectemp';
end
[~,index]=sort(score,'descend');
rank_image_ID=index(1:200,:);
QueryName=ref_list(rank_image_ID);
for j=1:200
    if j==200
        fprintf(fid,'%s\n',QueryName{j});
    else
        fprintf(fid,'%s ',QueryName{j});
    end
end
fprintf('%d %s\n',q,'th image processed');% ��ʾ���ڴ����·����ͼ����
end
fclose(fid);