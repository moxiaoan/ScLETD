N=260;
%Compute A 
A = zeros(N,N);
   
for k=1:2
    for l=1:2
        for i=1:N-1
            A(i,i,k,l) = -2;  A(i,i+1,k,l) = 1;  A(i+1,i,k,l) = 1;
        end
        A(N,N,k,l) = -2;
        if (k==1) 
            A(1,2,k,l) = 2;
        end
        if (l==1)
            A(N,N-1,k,l) = 2;
        end
    end
end

Px = zeros(N,N,2,2);
Pxi = zeros(N,N,2,2);
Dx = zeros(N,2,2);

for k=1:2
    for l=1:2      
        [Px(:,:,k,l),DDx] = eig(A(:,:,k,l));
        Pxi(:,:,k,l) = inv(Px(:,:,k,l));
        Dx(:,k,l) = DDx*ones(N,1);
    end
end
clear DDx DDy

for k=1:2
    for l=1:2
        id=(k-1)+(l-1)*2;
        v=Px(:,:,k,l);
        vname=strcat('v',sprintf('%d',id),sprintf('%d',N),'.dat');
        vi=Pxi(:,:,k,l);
        viname=strcat('vi',sprintf('%d',id),sprintf('%d',N),'.dat');
        DD=Dx(:,k,l);
        dname=strcat('d',sprintf('%d',id),sprintf('%d',N),'.dat');
        
        save(vname, 'v','-ASCII','-DOUBLE')  % Use when filename is stored in a variable
        save(viname, 'vi','-ASCII','-DOUBLE')  % Use when filename is stored in a variable
        save(dname, 'DD','-ASCII','-DOUBLE')  % Use when filename is stored in a variable
    end
end

   

%v=Px;
%vi=Pxi;

%save 'v512.dat' -ASCII -DOUBLE v
%save 'vi512.dat' -ASCII -DOUBLE vi
%save 'd512.dat' -ASCII -DOUBLE DD
