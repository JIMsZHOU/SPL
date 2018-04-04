clear
clc
load SDD.mat;

for i = 1:1:58509
    switch SDD_target(i,1)
        case 1
            SDD_target(i,:) = [1 0 0 0 0 0 0 0 0 0 0];
        case 2
            SDD_target(i,:) = [0 1 0 0 0 0 0 0 0 0 0];
        case 3
            SDD_target(i,:) = [0 0 1 0 0 0 0 0 0 0 0];
        case 4
            SDD_target(i,:) = [0 0 0 1 0 0 0 0 0 0 0];
        case 5
            SDD_target(i,:) = [0 0 0 0 1 0 0 0 0 0 0];
        case 6
            SDD_target(i,:) = [0 0 0 0 0 1 0 0 0 0 0];
        case 7
            SDD_target(i,:) = [0 0 0 0 0 0 1 0 0 0 0];
        case 8
            SDD_target(i,:) = [0 0 0 0 0 0 0 1 0 0 0];
        case 9
            SDD_target(i,:) = [0 0 0 0 0 0 0 0 1 0 0];
        case 10
            SDD_target(i,:) = [0 0 0 0 0 0 0 0 0 1 0];
        case 11
            SDD_target(i,:) = [0 0 0 0 0 0 0 0 0 0 1];

    end

end