Error = zeros(500,1);
for n = 100000 : -200 : 200
	H = 1 / n;
    for i = 1 : 100
	h1 = H*rand(1);
	h2 = H*rand(1);
	h3 = H*rand(1);
	a = 2*(2*h2 + h3)/h1/(h1+h2)/(h1+h2+h3);
	c = 2*(-h1 + h2 + h3) / h2 / h3 / (h1 + h2);
	d = 2*(h1-h2)/(h3)/(h2+h3)/(h1+h2+h3);
	b = -a-c-d;
	Error(n / 200) = Error(n / 200)+ abs(a * exp(1 - h1) + b * exp(1) + c*exp(1 + h2) + d*exp(1+h2+h3) -exp(1));
    end
    Error(n/200) = Error(n/200) / 100;
end
Error
plot(log(linspace(1/100000, 1/200, 500)'), log(Error))
A = [ones(500,1), log(linspace(1/100000, 1/200, 500)')];
Kp = (A'*A) \ (A'*log(Error));
