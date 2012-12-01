function xdot = cart_pole(x,u,sys0)

xd =  x(2,:);
xdd = 1./(sys0.mc+sys0.mp*sin(x(3,:)).^2).* ...
    (u + sys0.mp*sin(x(3,:)).*(sys0.l*x(4,:).^2 + sys0.g*cos(x(3,:))));
thd = x(4,:);
thdd = 1./(sys0.l*(sys0.mc+sys0.mp*sin(x(3,:)).^2)).* ...
    (-u.*cos(x(3,:)) - sys0.mp*sys0.l*x(4,:).^2.*cos(x(3,:)).*sin(x(3,:)) - ...
    (sys0.mc+sys0.mp)*sys0.g*sin(x(3,:)));

xdot = [xd ; xdd ; thd ; thdd];
