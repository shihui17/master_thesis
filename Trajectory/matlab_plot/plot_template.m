
clc;clear all;

% Daten laden, die geplottet werden sollen


%% Farben definieren

Farben.rot     =  [255 0 0 ]/255; %rot
Farben.blau    =  [0 0 255 ]/255; %blau
Farben.gruen   =  [0 128 0 ]/255; %grün
Farben.schwarz =  [0 0 0 ]/255; %schwarz
Farben.magenta =  [255 0 255 ]/255; %magenta
Farben.cyan    =  [0 255 255 ]/255; %cyan
Farben.orange  =  [255 180 0 ]/255; %orange
Farben.grau    =  [136 138 142 ]/255; %grau
Farben.hellrot =  [255 150 150 ]/255; %hellrot

Farben.imesblau   = [0 80 155 ]/255; %imesblau
Farben.imesorange = [231 123 41 ]/255; %imesorange
Farben.imesgruen  = [200 211 23 ]/255; %imesgrün

% Schriftart und -größe setzen
schrift.art     = 'Times'; %um die Schriftart zu ändern muss der string des Texts in z.B. \texfsf{...} geschrieben werden
schrift.groesse = 11;

% Figuregröße und Offsets für Ränder bzstart
% Abstand zwischen den Plots setzen
laengen.breite_fig           = 12;
laengen.hoehe_fig            = 18;
laengen.offset_breite_links  = 2;
laengen.offset_breite_rechts = 2;
laengen.offset_oben          = 1.5;
laengen.Abstand_plots        = 0.1;
laengen.hoehe_label          = 1.2;

% Figure aufrufen
f = figure(1);
clf(1);

set(f,'DefaultAxesUnit','centimeters')
set(f,'DefaultAxesFontName',schrift.art)
set(f,'DefaultAxesFontSize',schrift.groesse)
set(f,'DefaultAxesTickLabelInterpreter', 'latex')
set(f,'DefaultLegendInterpreter', 'latex')
set(f,'defaultTextInterpreter','latex')
set(f,'DefaultTextFontSize',schrift.groesse)


f.Units             = 'centimeters';
f.OuterPosition  	= [30 5 laengen.breite_fig laengen.hoehe_fig];
f.Color             = [1 1 1];
f.PaperSize         = [laengen.breite_fig laengen.hoehe_fig];
f.PaperPosition     = [0 0 0 0];
f.ToolBar           = 'none';
f.MenuBar           = 'none';


%Anzahl der Zeilen
n=3;

%Anzahl labels in x 
m=2;

%Figureposition setzen
laengen.breite_axes    = laengen.breite_fig - (laengen.offset_breite_links+laengen.offset_breite_rechts);
laengen.hoehe_axes     = (laengen.hoehe_fig-laengen.offset_oben-(n-1)*laengen.Abstand_plots-m*laengen.hoehe_label)/n;

positionen(1).pos      = [laengen.offset_breite_links   laengen.hoehe_fig-(laengen.offset_oben + laengen.hoehe_axes)  laengen.breite_axes     laengen.hoehe_axes];
positionen(2).pos      = positionen(1).pos + [0   -(laengen.Abstand_plots + laengen.hoehe_axes)     0     0];
positionen(3).pos      = positionen(2).pos + [0   -(laengen.Abstand_plots + laengen.hoehe_axes + laengen.hoehe_label)     0     0];

%% Plot 1

achsen(1).a = axes;
achsen(1).a.Position    = positionen(1).pos; 

plots(1).p1            = plot(data.x,data.y1);
plots(1).p1.LineStyle  = '-';
plots(1).p1.LineWidth  = 1.5;
plots(1).p1.Color      = Farben.imesblau;
hold on
plots(1).p2            = plot(2*data.x,data.y1);
plots(1).p2.LineStyle  = '--';
plots(1).p2.LineWidth  = 1.5;
plots(1).p2.Color      = Farben.imesorange;

limits.ymin = -1.2;
limits.ymax = 1.2;    
limits.xmin = 0;
limits.xmax = 5*pi;

achsen(1).a.XLim        = [limits.xmin limits.xmax];
achsen(1).a.YLim        = [limits.ymin limits.ymax];
achsen(1).a.XGrid       = 'on';
achsen(1).a.YGrid       = 'on';
achsen(1).a.XTickLabel  = [];

label(1).y            = ylabel('Beschleunigung $a_x$ in $\frac{\mathrm{m}}{\mathrm{s}^2}$');
%% Plot 2

achsen(2).a = axes;

achsen(2).a.Position   = positionen(2).pos;   
plots(2).p1            = plot(data.x,data.y1);
plots(2).p1.LineStyle  = '-';
plots(2).p1.LineWidth  = 1.5;
plots(2).p1.Color      = Farben.imesblau;
hold on
plots(2).p2            = plot(2*data.x,data.y2);
plots(2).p2.LineStyle  = '--';
plots(2).p2.LineWidth  = 1.5;
plots(2).p2.Color      = Farben.imesorange;




limits.ymin            = -1.2;
limits.ymax            = 1.2;  

achsen(2).a.XLim        = [limits.xmin limits.xmax];
achsen(2).a.YLim        = [limits.ymin limits.ymax];

achsen(2).a.XTick       = [0 5 10 15 20 25 30];
label(2).x              = xlabel('Zeit $t$ in $\mathrm{s}$');
label(2).y              = ylabel('Winkel $\alpha_{\mathrm{test}}$ in $^{\circ}$');
label(2).y.Units        = 'centimeter';


% zweite y-Achse

yyaxis(achsen(2).a,'right');
achsen(2).a.YColor        = Farben.rot;
plots(2).p1r              = plot(data.x,10*data.y1);
plots(2).p1r.LineStyle    = ':';
plots(2).p1r.LineWidth    = 1.5;
plots(2).p1r.Color        = Farben.rot;


limits.ymin             = -11;
limits.ymax             = 11;  

achsen(2).a.XLim        = [limits.xmin limits.xmax];
achsen(2).a.YLim        = [limits.ymin limits.ymax];

achsen(2).a.XTick       = [0 5 10 15 20 25 30];
label(2).yr             = ylabel({'Geschwindigkeit $\centering v$ in $\frac{\mathrm{m}}{\mathrm{s}}$'});
label(2).yr.Units       = 'centimeter';
achsen(2).a.YTickLabel  = strrep(achsen(2).a.YTickLabel,'.',',');


%% Plot 3

achsen(3).a = axes;
achsen(3).a.Position   = positionen(3).pos;   
plots(3).p1            = plot(data.x,data.y1);
plots(3).p1.LineStyle  = '-';
plots(3).p1.LineWidth  = 1.5;
plots(3).p1.Color      = Farben.imesblau;

hold on
plots(3).p2            = plot(2*data.x,data.y1);
plots(3).p2.LineStyle  = '--';
plots(3).p2.LineWidth  = 1.5;
plots(3).p2.Color      = Farben.imesorange;

plots(3).p3            = plot(data.x,data.y3);
plots(3).p3.LineStyle  = '-.';
plots(3).p3.LineWidth  = 1.5;
plots(3).p3.Color      = Farben.imesgruen;


y_noise                = data.y3+0.15*randn(size(data.y3));


plots(3).p4            = plot(data.x(1:10:end),y_noise(1:10:end));
plots(3).p4.Marker     = 'x';
plots(3).p4.MarkerSize = 2.5;
plots(3).p4.LineStyle  = 'none';
plots(3).p4.Color      = Farben.imesgruen/2;

limits.ymin            = -1.5;
limits.ymax            = 5; 

achsen(3).a.XLim        = [limits.xmin limits.xmax];
achsen(3).a.YLim        = [limits.ymin limits.ymax];

achsen(3).a.XTick       = [0 5 10 15 20 25 30];
label(3).x              = xlabel('Position $p$ in $\mathrm{m}$');
label(3).y              = ylabel('Drehrate $\dot{\psi}$ in $\frac{^{\circ}}{\mathrm{s}}$');
label(3).y.Position(1)  = label(1).y.Position(1);
achsen(3).a.YTickLabel  = strrep(achsen(3).a.YTickLabel,'-1','$\dot{\psi}_{\mathrm{min}}$');


%% Plot 4 (Zoom in Plot 3)

limits.xmin          = 9.5;
limits.xmax          = 10;
limits.ymin          = -1.2; 
limits.ymax          = 0.4;

plots(5).box             = plot([limits.xmin,limits.xmin,limits.xmax,limits.xmax,limits.xmin],[limits.ymin,limits.ymax,limits.ymax,limits.ymin,limits.ymin]);
plots(5).box.Color       = 'k';
plots(5).box.LineWidth   = 1;

achsen(4).a = axes;
achsen(4).a.Position    = positionen(3).pos+[1.5 2.5 0 0];   
achsen(4).a.Position(3) = 2.5;
achsen(4).a.Position(4) = 2;
plots(4).p1            = plot(data.x,data.y1);
plots(4).p1.LineStyle  = '-';
plots(4).p1.LineWidth  = 1.5;
plots(4).p1.Color      = Farben.imesblau;

hold on
plots(4).p2            = plot(2*data.x,data.y1);
plots(4).p2.LineStyle  = '--';
plots(4).p2.LineWidth  = 1.5;
plots(4).p2.Color      = Farben.imesorange;

plots(4).p3            = plot(data.x,data.y3);
plots(4).p3.LineStyle  = '-.';
plots(4).p3.LineWidth  = 1.5;
plots(4).p3.Color      = Farben.imesgruen;

plots(4).p4            = plot(data.x,y_noise);
plots(4).p4.Marker     = plots(3).p4.Marker;
plots(4).p4.MarkerSize = plots(3).p4.MarkerSize;
plots(4).p4.LineStyle  = 'none';
plots(4).p4.Color      = plots(3).p4.Color;

limits.ymin = -1.2;
limits.ymax = 0.4;    
achsen(4).a.XLim        = [limits.xmin limits.xmax];
achsen(4).a.YLim        = [limits.ymin limits.ymax];

achsen(4).a.XTick       = [limits.xmin limits.xmax];


legende(1).l             = legend([plots(3).p1,plots(3).p2, plots(3).p3, plots(3).p4],'Modell 1','Modell 2','Modell 3','Messung');
legende(1).l.Units       = 'centimeters';
legende(1).l.Position    = [8.5 4.8 0 0];
legende(1).l.Box         = 'off';


% Text hinzufügen
annotationen(1).text             = annotation(f,'textbox');
annotationen(1).text.Units       = 'centimeters';
annotationen(1).text.Position    = [3.2 11.5 0.1 0.1];
annotationen(1).text.Color       = [1 1 1]/2;
annotationen(1).text.String      = 'Beispieltext';
annotationen(1).text.Interpreter = 'Latex';
annotationen(1).text.EdgeColor   = 'none';
annotationen(1).text.FontSize    = 12;


% Pfeil hinzufügen
annotationen(2).pfeil             = annotation(f,'arrow');
annotationen(2).pfeil.Units       = 'centimeters';
annotationen(2).pfeil.Position    = [4 9 0.5 2];


%% Speichern und exportieren
filename = 'template';
set(gcf,'PaperPositionMode','auto');
print('-depsc',[filename,'_tmp.eps']); %Ausgabe als .eps
system(['gswin32c.exe -dNOPAUSE -dBATCH -dNOCACHE -dEPSCrop -sDEVICE=epswrite -sOutputFile=',filename,'.eps ',filename,'_tmp.eps']);
system(['gswin32c.exe -dNOPAUSE -dBATCH -dNOCACHE -dEPSCrop -sDEVICE=pdfwrite -sOutputFile=',filename,'.pdf ',filename,'_tmp.eps']);
system(['gswin32c.exe -dNOPAUSE -dBATCH -dNOCACHE -r300 -dEPSCrop -sDEVICE=png16m   -sOutputFile=',filename,'.png ',filename,'_tmp.eps']);

