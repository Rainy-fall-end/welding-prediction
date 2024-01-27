clear;clc;

load('para_new.mat')
%%
num=15;
x=0:10:200;
y=0:10:400;
p_num=numel(x)*numel(y);
for i=1:numel(x)
    for j=1:numel(y)
        xy(j+numel(y)*(i-1),1)=x(i);
        xy(j+numel(y)*(i-1),2)=y(j);
    end
end
load('data2_new.mat')
for i=1:20
    ii=num2str(i);
    da=eval(['e',ii]);
    da(:,1)=da(:,1)-0.5*para(i,2);
    k=dsearchn(da(:,1:2),xy(1:p_num,:));
    Y0(i,1:p_num)=da(k,3)+da(k,4)-6;
end
X0=(para-ones(20,2).*min(para))./(max(para)-min(para));
X=X0;
Y=Y0;
%% initial
pxy=[0 0; 0 200; 0 400;100 0;100 200;100 400;200 0;200 200;200 400];
kk=dsearchn(xy,pxy);
dmax=numel(pxy(:,1));
% 归一化
pxy(:,1)=(pxy(:,1))./200;
pxy(:,2)=(pxy(:,2))./400;
xy(:,1)=(xy(:,1))./200;
xy(:,2)=(xy(:,2))./400;

theta0=[10 10 ];
lob =[1e-20 1e-20 ]; 
upb = [40 40];
for k=1:dmax
    X1=X(1:14,1:2);
    indices=crossvalind('Kfold',14,14);
    for i=1:14
        test=(indices==i);
        train=~test;
        train_data=X1(train,:);
        test_data=X1(test,:);
        train_target=Y(train,kk(k));
        test_target=Y(test,kk(k));
        [dmodel] =dacefit(train_data, train_target, @regpoly1, @corrgauss, theta0, lob, upb);
        [YEE([1:14,num],i),~]=predictor(X([1:14,num],1:2),dmodel);
    end
    ans=mean(YEE,2);
    YE(1:15,kk(k))=ans;
    err1(:,k)=abs(Y([1:14,num],kk(k))-YE(:,kk(k)));
end
theta1=[10 10] ;
lob1 =[1e-20 1e-20]; 
upb1 = [40 40] ;

p_dis=YE(15,kk)';
[dmodel2_g]=dacefit(pxy,p_dis, @regpoly1, @corrgauss, theta1, lob1, upb1);
[pred_ini,~]=predictor(xy,dmodel2_g);
d=Y0(num,:)';
err_ini=abs(d-pred_ini);
err_ini_max2=max(err_ini);
[r2_ini,rmse_ini]=rsquare(d,pred_ini);

xy2=xy;
pred_op=pred_ini;
pxy2=pxy;
kk2=kk;
p_dis2=p_dis;

%%
err_p=abs(d-pred_op);
err_p_max=max(err_p);
err_mean=mean(err_p);
%err_p_max2=max(err_p(1:820));
[r2_p,rmse_p]=rsquare(pred_op,d);

xy2(:,1)=xy2(:,1)*200;
xy2(:,2)=xy2(:,2)*400;
pxy2(:,1)=pxy2(:,1)*200;
pxy2(:,2)=pxy2(:,2)*400;
%[X,Y,Z]=griddata(xy(:,1),xy(:,2),d,linspace(min(xy(:,1)),max(xy(:,1)),21)',linspace(min(xy(:,2)),max(xy(:,2)),41),'v4');
[X1,Y1,Z1]=griddata(xy2(:,1),xy2(:,2),pred_ini,linspace(min(xy2(:,1)),max(xy2(:,1)),21)',linspace(min(xy2(:,2)),max(xy2(:,2)),41),'v4');
[X4,Y4,Z4]=griddata(xy2(:,1),xy2(:,2),d,linspace(min(xy2(:,1)),max(xy2(:,1)),21)',linspace(min(xy2(:,2)),max(xy2(:,2)),41),'v4');
[X3,Y3,Z3]=griddata(xy2(:,1),xy2(:,2),pred_op,linspace(min(xy2(:,1)),max(xy2(:,1)),21)',linspace(min(xy2(:,2)),max(xy2(:,2)),41),'v4');
figure(1)
surf(X1,Y1,Z1);
%hold on
figure(2)
surf(X4,Y4,Z4);
%shading interp;
%hold on
figure(3)
surf(X3,Y3,Z3);
xlabel({'X (mm)'});
ylabel({'Y (mm)'});
zlabel({'Z distortion (mm)'});
figure(4)
plot(pxy2(1:dmax,1),pxy2(1:dmax,2),'r*')
hold on
plot(pxy2(dmax+1:end,1),pxy2(dmax+1:end,2),'b*')
hold on
scatter(xy2(:,1),xy2(:,2),5,'filled','g')
xlim([-50,250]);
ylim([-50,450]);
xlabel({'X (mm)'});
ylabel({'Y (mm)'});