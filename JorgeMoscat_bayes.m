% Jorge Moscat 
%Assigment 5 - Naives Bayes Classifier for Iris Data
%
%This progam will just ouput the confussion matrices for the 9
%classifiers. I took those outputs of a sample run of the code and pasted
%the confussion matrices in the attached spreadsheet for analysis.
%
%
%NOTE: Column 5 contains the actual classification. If it is
% =1 ---> Setosa
% =2 ---> Versicolor
% =3 ---> Virginica




load iris.txt;
%Iris data will be discretize in 7 buckets
buckets=7;

%This matrices will hold the likelihood probabilites -->e.g P(Atr1=2|Setosa)
likelihood_setosa = zeros(buckets,4);
likelihood_versicol = zeros(buckets, 4);
likelihood_virg = zeros(buckets, 4); 

%Prior probabilities 
prior_setosa = 0;
prior_versicol = 0;
prior_virg = 0;

%discretazing the data. 
[n1,x1]=hist(iris(:,1),buckets);
for i=1:150, for j=1:buckets, d(i,j)=(iris(i,1)-x1(j))^2; end; end
for i=1:150, temp=find(d(i,:)== min(d(i,:))); iris2(i,1)=temp(1); end

[n2,x2]=hist(iris(:,2),buckets);
for i=1:150, for j=1:buckets, d2(i,j)=(iris(i,2)-x2(j))^2; end; end
for i=1:150, temp=find(d2(i,:)== min(d2(i,:))); iris2(i,2)=temp(1); end

[n3,x3]=hist(iris(:,3),buckets);
for i=1:150, for j=1:buckets, d3(i,j)=(iris(i,3)-x3(j))^2; end; end
for i=1:150, temp=find(d3(i,:)== min(d3(i,:))); iris2(i,3)=temp(1); end
 
[n4,x4]=hist(iris(:,4),buckets);
for i=1:150, for j=1:buckets, d4(i,j)=(iris(i,4)-x4(j))^2; end; end
for i=1:150, temp=find(d4(i,:)== min(d4(i,:))); iris2(i,4)=temp(1); end

%add classification column to discretized matrix
iris2(:,5)=iris(:,5);

%Get sample for training


p = randperm(150);
%sampleindex=randint(1,15,150);

for iterations=1:9
    temp = 15*iterations;
    training = iris2(p(1,1:temp),:);
    temp2 = temp+1;
    test = iris2(p(1,temp2:150),:);

    %------Setosa Classification -----

    [setosa_count,~] = size(find (training(:,5) == 1));
    prior_setosa = setosa_count/15; %Estimating P(Setosa)

    for attribute=1:4
        for values=1:buckets
            %Estimating P(a|setosa) for each attribute value of each attribute
            [count, ~] = size(find (training(:,attribute)== values & training(:,5) == 1));
            likelihood_setosa (values, attribute) = count/setosa_count;
        end
    end

    %------Versicolor Classification -----

    [versicolor_count,~] = size(find (training(:,5) == 2));
    prior_versicol = versicolor_count/15; 

    for attribute=1:4
        for values=1:buckets
            [count, ~] = size(find (training(:,attribute)== values & training(:,5) == 2));
            likelihood_versicol (values, attribute) = count/versicolor_count;
        end
    end

    %------Virginica Classification -----

    [virginica_count,~] = size(find (training(:,5) == 3));
    prior_virg = virginica_count/15; 

    for attribute=1:4
        for values=1:buckets
            [count, ~] = size(find (training(:,attribute)== values & training(:,5) == 3));
            likelihood_virg (values, attribute) = count/virginica_count;
        end
    end






    %Classify examples (15-150)

    [size_test,~] = size(test);
    predicted_results = zeros(size_test,1);
    for i=1:size_test
        prob_its_setosa = prior_setosa*likelihood_setosa(test(i,1),1)*likelihood_setosa(test(i,2),2)*likelihood_setosa(test(i,3),3)*likelihood_setosa(test(i,4),4);
        prob_its_versicolor = prior_versicol*likelihood_versicol(test(i,1),1)*likelihood_versicol(test(i,2),2)*likelihood_versicol(test(i,3),3)*likelihood_versicol(test(i,4),4);
        prob_its_virginica = prior_virg*likelihood_virg(test(i,1),1)*likelihood_virg(test(i,2),2)*likelihood_virg(test(i,3),3)*likelihood_virg(test(i,4),4);
        [~,predicted_class] = max([prob_its_setosa prob_its_versicolor prob_its_virginica]);
        predicted_results(i,1) = predicted_class;
    end
    
    %Creating and displaying confusion matrix
    [C,order] = confusionmat(test(:,5), predicted_results);
    disp(C);
end


