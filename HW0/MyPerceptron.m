function w = MyPerceptron(X, y, w0)
    w =  w0;
    for i = 1:40
        if (dot(w, X(i,:)) * y(i)) <= 0
            w = w + y(i) * X(i,:);
        end
    end
    
    subplot(1,2,1); 
    axis([-1 1 -1 1]);
    hold on;
    
    for i = 1:40
        if y(i) == 1
            scatter(X(i,1), X(i,2), [], [1, 0, 0], '*');
        else
            scatter(X(i,1), X(i,2), [], [0, 0, 1], '*');
        end
    end
    line(X(:,1), (-w0(1) * X(:,1)) / w0(2));
    
    
    subplot(1,2,2);
    axis([-1 1 -1 1]);
    hold on;
    for i = 1:40
        if y(i) == 1
            scatter(X(i,1), X(i,2), [], [1, 0, 0], '*');
        else
            scatter(X(i,1), X(i,2), [], [0, 0, 1], '*');
        end
    end
    line(X(:,1), (-w(1) * X(:,1)) / w(2));
