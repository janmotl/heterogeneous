% Empirically estimate difference between two samples.

% expected absolute difference between normally distributed features is 2/sqrt(pi) ~= 1.1284.
% See: https://en.wikipedia.org/wiki/Mean_absolute_difference#Examples
nrow = 200;
ncol = 20;
x=randn(nrow, ncol);

acc=0;
for i=1:nrow
    for j=i+1:nrow
        acc = acc+sum(abs(x(i,:)-x(j,:)));
    end
end
avg = acc / (nrow*(nrow-1)/2) / ncol

%% expected squared difference between normally distributed features is 2.
nrow = 200;
ncol = 20;
x=randn(nrow, ncol);

acc=0;
for i=1:nrow
    for j=i+1:nrow
        acc = acc + sum( (x(i,:)-x(j,:)).^2 );
    end
end
avg = acc / (nrow*(nrow-1)/2) / ncol

%% expected absolute difference between binary features is 0.5.
nrow = 200;
ncol = 20;
x = rand(nrow, ncol)>0.5;

acc=0;
for i=1:nrow
    for j=i+1:nrow
        acc = acc+sum(abs(x(i,:)-x(j,:)));
    end
end
avg = acc / (nrow*(nrow-1)/2) / size(x,2)

%% expected difference between polynomial features is 1 - 1/cardinality.
nrow = 200;
ncol = 20;
x = randi(4, nrow, ncol);

acc=0;
for i=1:nrow
    for j=i+1:nrow
        acc = acc+sum(x(i,:) ~= x(j,:));
    end
end
avg = acc / (nrow*(nrow-1)/2) / size(x,2)