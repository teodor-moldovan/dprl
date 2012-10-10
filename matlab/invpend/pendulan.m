function [sys,x0,str,ts] = pendulan(t,x,u,flag,xinit,xlimit,plen,fsat,samplet)
%PENDULAN S-function for making pendulum animation.
%
%   $Revision: 1.2 $

% Plots every major integration step, but has no states of its own
global csv

switch flag,

  %%%%%%%%%%%%%%%%%%
  % Initialization %
  %%%%%%%%%%%%%%%%%%
  case 0,
    [sys,x0,str,ts]=mdlInitializeSizes(xinit,xlimit,plen,samplet);

  %%%%%%%%%%
  % Update %
  %%%%%%%%%%
  case 2,
    sys=mdlUpdate(t,x,u,plen,fsat);

  %%%%%%%%%%%%%%%%
  % Unused flags %
  %%%%%%%%%%%%%%%%
  case { 1, 3, 4, 9 },
    sys = [];
    
  %%%%%%%%%%%%%%%
  % DeleteBlock %
  %%%%%%%%%%%%%%%
  case 'DeleteBlock',
    LocalDeleteBlock
    
  %%%%%%%%%%%%%%%
  % DeleteFigure %
  %%%%%%%%%%%%%%%
  case 'DeleteFigure',
    LocalDeleteFigure
  
  %%%%%%%%%%
  % Slider %
  %%%%%%%%%%
% $$$   case 'Slider',
% $$$     LocalSlider
  
  %%%%%%%%%
  % Close %
  %%%%%%%%%
  case 'Close',
    LocalClose
     
  %%%%%%%%%%%%%%%%%%%%
  % Unexpected flags %
  %%%%%%%%%%%%%%%%%%%%
  otherwise
    error(['Unhandled flag = ',num2str(flag)]);
end

% end pendelanan

%
%=============================================================================
% mdlInitializeSizes
% Return the sizes, initial conditions, and sample times for the S-function.
%=============================================================================
%
function [sys,x0,str,ts]=mdlInitializeSizes(xinit,xlimit,plen,samplet)

%
% call simsizes for a sizes structure, fill it in and convert it to a
% sizes array.
%
sizes = simsizes;

sizes.NumContStates  = 0;
sizes.NumDiscStates  = 0;
sizes.NumOutputs     = 0;
sizes.NumInputs      = 6;
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;

sys = simsizes(sizes);

%
% initialize the initial conditions
%
x0  = [];

%
% str is always an empty matrix
%
str = [];

%
% initialize the array of sample times, for the pendulum demo,
% the animation is updated every 0.1 seconds
%
ts  = [samplet 0];

%
% create the figure, if necessary
%
LocalPendInit(xinit,xlimit,plen);

% end mdlInitializeSizes

%
%=============================================================================
% mdlUpdate
% Update the pendulum animation.
%=============================================================================
%
function sys=mdlUpdate(t,x,u,plen,fsat)

fig = get_param(gcbh,'UserData');
if ishandle(fig),
  if strcmp(get(fig,'Visible'),'on'),
    ud = get(fig,'UserData');
    LocalPendSets(t,ud,u,plen,fsat);
  end
end;
 
sys = [];

% end mdlUpdate

%
%=============================================================================
% LocalDeleteBlock
% The animation block is being deleted, delete the associated figure.
%=============================================================================
%
function LocalDeleteBlock

fig = get_param(gcbh,'UserData');
if ishandle(fig),
  delete(fig);
  set_param(gcbh,'UserData',-1)
end

% end LocalDeleteBlock

%
%=============================================================================
% LocalDeleteFigure
% The animation figure is being deleted, set the S-function UserData to -1.
%=============================================================================
%
function LocalDeleteFigure

ud = get(gcbf,'UserData');
set_param(ud.Block,'UserData',-1);
  
% end LocalDeleteFigure

%
%=============================================================================
% LocalSlider
% The callback function for the animation window slider uicontrol.  Change
% the reference block's value.
%=============================================================================
%
% $$$ function LocalSlider
% $$$ 
% $$$ ud = get(gcbf,'UserData');
% $$$ set_param(ud.RefBlock,'Value',num2str(get(gcbo,'Value')));

% end LocalSlider

%
%=============================================================================
% LocalClose
% The callback function for the animation window close button.  Delete
% the animation figure window.
%=============================================================================
%
function LocalClose

delete(gcbf)

% end LocalClose


%
%=============================================================================
% LocalPendSets
% Local function to set the position of the graphics objects in the
% inverted pendulum animation window.
%=============================================================================
%
function LocalPendSets(time,ud,u,plen,fsat)

global csv

XDelta   = 0.05;
PDelta   = 0.005;
XPendTop = u(2) + plen*sin(u(3));
YPendTop = plen*cos(u(3));
PDcosT   = PDelta*cos(u(3));
PDsinT   = -PDelta*sin(u(3));
set(ud.Pend,...
  'XData',[XPendTop-PDcosT XPendTop+PDcosT; u(2)-PDcosT u(2)+PDcosT], ...
  'YData',[YPendTop-PDsinT YPendTop+PDsinT; -PDsinT PDsinT]);
set(ud.TimeField,...
  'String',num2str(time));
if u(4)==0 & ~strcmp(get(ud.MotorState,'String'),'Off'), 
  set(ud.MotorState,'String','Off');
  set(ud.Cart,...
      'XData',ones(2,1)*[u(2)-XDelta u(2)+XDelta]);
  set(ud.Arrow,...
      'Visible','off');
  if u(1)~=0,
    if isunix
%       fp=fopen('/dev/audio','wb');
%       fwrite(fp,lin2mu(csv),'uchar');
    else
%       sound(csv);
    end
  end
elseif u(4) ~= 0,
  set(ud.MotorState,'String','On');
  set(ud.Cart,...
      'XData',ones(2,1)*[u(2)-XDelta u(2)+XDelta]);
  if abs(u(5)) > 0.005*fsat & u(6) ~= 0,
    % show the error as long as the input is greater than 0.5 % of
    % the saturation force
    set(ud.Arrow,...
      'XData',ones(2,1)*(u(5)*[0 -0.5*XDelta/abs(u(5)) -0.5*XDelta/abs(u(5)) ...
             -0.5*XDelta/abs(u(5))-0.5*5*XDelta/fsat]+u(2)-sign(u(5))*XDelta),...
      'Visible','on');
  else
    set(ud.Arrow,'Visible','off');
  end
end
% $$$ set(ud.RefMark,...
% $$$   'XData',u(1)+[-XDelta 0 XDelta]);

% Force plot to be drawn
% pause(0)
drawnow

% end LocalPendSets

%
%=============================================================================
% LocalPendInit
% Local function to initialize the pendulum animation.  If the animation
% window already exists, it is brought to the front.  Otherwise, a new
% figure window is created.
%=============================================================================
%
function LocalPendInit(xinit,xlimit,plen)

%
% The name of the reference is derived from the name of the
% subsystem block that owns the pendulum animation S-function block.
% This subsystem is the current system and is assumed to be the same
% layer at which the reference block resides.
%
sys = get_param(gcs,'Parent');

TimeClock = 0;
RefSignal = 0; % str2num(get_param([sys '/' RefBlock],'Value'));
XCart     = xinit(1);
Theta     = xinit(3);

XDelta    = 0.05;
PDelta    = 0.005;
XPendTop  = XCart + plen*sin(Theta); % Will be zero
YPendTop  = plen*cos(Theta);         % Will be 10
PDcosT    = PDelta*cos(Theta);     % Will be 0.2
PDsinT    = -PDelta*sin(Theta);    % Will be zero

%
% The animation figure handle is stored in the pendulum block's UserData.
% If it exists, initialize the reference mark, time, cart, and pendulum
% positions/strings/etc.
%
Fig = get_param(gcbh,'UserData');
if ishandle(Fig),
  FigUD = get(Fig,'UserData');
  set(FigUD.RefMark,...
       'XData',RefSignal+[-XDelta 0 XDelta]);
  set(FigUD.TimeField,...
      'String',num2str(TimeClock));
  set(FigUD.Cart,...
      'XData',ones(2,1)*[XCart-XDelta XCart+XDelta]);
  set(FigUD.Pend,...
      'XData',[XPendTop-PDcosT XPendTop+PDcosT; XCart-PDcosT XCart+PDcosT],...
      'YData',[YPendTop-PDsinT YPendTop+PDsinT; -PDsinT PDsinT]);
      
  %
  % bring it to the front
  %
  figure(Fig);
  pause(1);
  return
end

%
% the animation figure doesn't exist, create a new one and store its
% handle in the animation block's UserData
%
FigureName = 'Pendulum Visualization';
Fig = figure(...
  'Units',           'normalized',...
  'Name',            FigureName,...
  'NumberTitle',     'off',...
  'MenuBar',         'none',...
  'IntegerHandle',   'off',...
  'HandleVisibility','callback',...
  'Resize',          'on',...
  'DeleteFcn',       'pendulan([],[],[],''DeleteFigure'')',...
  'CloseRequestFcn', 'pendulan([],[],[],''Close'');');
AxesH = axes(...
  'Parent',  Fig,...
  'Units',   'normalized',...
  'Position',[0.1 0.15 0.8 0.82],...
  'CLim',    [1 64], ...
  'Xlim',    [-0.6 0.6],...
  'XTick',    [],...
  'Ylim',    [-0.3 0.8],...
  'YTick',    [],...
  'Visible', 'on');
axes(...
  'Parent',  Fig,...
  'Units',   'normalized',...
  'Position',[0.1 0.29 0.8 0.01],...
  'CLim',    [1 64], ...
  'Xlim',    [-0.6 0.6],...
  'Ylim',    [0 0.1],...
  'YTick',    [],...
  'Visible', 'on');
line([-0.6 0.6],[-0.1 -0.1],'Parent',  AxesH,'LineWidth',2,'Color','k')
Cart = surface(...
  'Parent',   AxesH,...
  'XData',    ones(2,1)*[XCart-XDelta XCart+XDelta],...
  'YData',    [0 0; -0.1 -0.1],...
  'ZData',    zeros(2),...
  'CData',    ones(2),...
  'EraseMode','xor');
Pend = surface(...
  'Parent',   AxesH,...
  'XData',    [XPendTop-PDcosT XPendTop+PDcosT; XCart-PDcosT XCart+PDcosT],...
  'YData',    [YPendTop-PDsinT YPendTop+PDsinT; -PDsinT PDsinT],...
  'ZData',    zeros(2),...
  'CData',    11*ones(2),...
  'EraseMode','xor');
Arrow = surface(...
  'Parent',   AxesH,...
  'XData',    ones(2,1)*[XCart-XDelta XCart-XDelta-0.05 XCart-XDelta-0.05 XCart-XDelta-0.1],...
  'YData',    0.5*[0 0.025 0.01 0.01;0 -0.025 -0.01 -0.01]-0.05,...
  'ZData',    zeros(2,4),...
  'CData',    ones(2,4),...
  'Visible',  'off',...
  'EraseMode','xor');
RefMark = patch(...
   'Parent',   AxesH,...
   'XData',    RefSignal+[-XDelta 0 XDelta],...
   'YData',    [-2 -2 -2],...
   'CData',    22);
Endpointp = patch(...
   'Parent',   AxesH,...
   'XData',    xlimit+XDelta+[0 0.005 0.01],...
   'YData',    [-0.1 0 -0.1],...
   'CData',    22);
Endpointm = patch(...
   'Parent',   AxesH,...
   'XData',    -xlimit-XDelta+[-0.01 -0.005 0],...
   'YData',    [-0.1 0 -0.1],...
   'CData',    22);
uicontrol(...
  'Parent',  Fig,...
  'Style',   'text',...
  'Units',   'normalized',...
  'Position',[0 0 1 0.1]);
uicontrol(...
  'Parent',             Fig,...
  'Style',              'text',...
  'Units',              'normalized',...
  'Position',           [0.5 0.01 0.2 0.05], ...
  'HorizontalAlignment','right',...
  'String',             'Time: ');
TimeField = uicontrol(...
  'Parent',             Fig,...
  'Style',              'text',...
  'Units',              'normalized', ...
  'Position',           [0.7 0.01 0.1 0.05],...
  'HorizontalAlignment','left',...
  'String',             num2str(TimeClock));
% $$$ SlideControl = uicontrol(...
% $$$   'Parent',   Fig,...
% $$$   'Style',    'slider',...
% $$$   'Units',    'pixel', ...
% $$$   'Position', [100 25 300 22],...
% $$$   'Min',      -9,...
% $$$   'Max',      9,...
% $$$   'Value',    RefSignal,...
% $$$   'Callback', 'pendan([],[],[],''Slider'');');
uicontrol(...
  'Parent',  Fig,...
  'Style',   'pushbutton',...
  'Units',              'normalized', ...    
  'Position',[0.83 0.01 0.14 0.06],...
  'String',  'Close', ...
  'Callback','pendulan([],[],[],''Close'');');
uicontrol(...
  'Parent',             Fig,...
  'Style',              'text',...
  'Units',              'normalized',...
  'Position',           [0.04 0.01 0.1 0.05], ...
  'HorizontalAlignment','left',...
  'String',             'Motor:');
MotorState = uicontrol(...
  'Parent',             Fig,...
  'Style',              'text',...
  'Units',              'normalized',...
  'Position',           [0.12 0.01 0.2 0.05], ...
  'HorizontalAlignment','left',...
  'String',             'Off');
set(RefMark,'EraseMode','xor');

%
% all the HG objects are created, store them into the Figure's UserData
%
FigUD.Cart         = Cart;
FigUD.Pend         = Pend;
FigUD.Pendlength   = plen;
FigUD.TimeField    = TimeField;
% $$$ FigUD.SlideControl = SlideControl;
FigUD.RefMark      = RefMark;
FigUD.Arrow        = Arrow;
FigUD.Block        = get_param(gcbh,'Handle');
FigUD.MotorState   = MotorState;
%FigUD.RefBlock     = get_param([sys '/' RefBlock],'Handle');
set(Fig,'UserData',FigUD);

drawnow
pause(1);

%
% store the figure handle in the animation block's UserData
%
set_param(gcbh,'UserData',Fig);

% end LocalPendInit
