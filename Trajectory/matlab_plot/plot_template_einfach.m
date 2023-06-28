
clc;clear all;



% Daten laden, die geplottet werden sollen
load('data');



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

% Figuregröße und Offsets für Ränder bzw. Abstand zwischen den Plots setzen
laengen.breite_fig           = 12;
laengen.hoehe_fig            = 8;

laengen.offset_breite_links  = 2;
laengen.offset_breite_rechts = 0.5;
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
n=1;

%Anzahl labels in x 
m=1;

%Figureposition setzen
laengen.breite_axes    = laengen.breite_fig - (laengen.offset_breite_links+laengen.offset_breite_rechts);
laengen.hoehe_axes     = (laengen.hoehe_fig-laengen.offset_oben-(n-1)*laengen.Abstand_plots-m*laengen.hoehe_label)/n;

positionen(1).pos      = [laengen.offset_breite_links   laengen.hoehe_fig-(laengen.offset_oben + laengen.hoehe_axes)  laengen.breite_axes     laengen.hoehe_axes];

%% Plot 1

achsen(1).a = axes;
achsen(1).a.Position   = positionen(1).pos; 

plots(1).p1            = plot(data.x,data.y(1, :));
plots(1).p1.LineStyle  = '-';
plots(1).p1.LineWidth  = 1.5;
plots(1).p1.Color      = Farben.imesblau;
hold on
%plots(1).p2            = plot(2*data.x,data.y1);
%plots(1).p2.LineStyle  = '--';
%plots(1).p2.LineWidth  = 1.5;
%plots(1).p2.Color      = Farben.imesorange;

limits.ymin            = min(data.y(1, 1));
limits.ymax            = max(data.y(1, :));    
limits.xmin            = 0;
limits.xmax            = data.x(end);

achsen(1).a.XLim        = [limits.xmin limits.xmax];
achsen(1).a.YLim        = [limits.ymin limits.ymax];
achsen(1).a.XGrid       = 'on';
achsen(1).a.YGrid       = 'on';

label(1).y              = ylabel('Joint angle $a_x$ in $\frac{\mathrm{m}}{\mathrm{s}^2}$');
label(1).x              = xlabel('Zeit $t$ in $\mathrm{s}$');


%legenden(1).l            = legend([plots(1).p1,plots(1).p2],'Modell 1','Modell 2');
%legenden(1).l.Units      = 'centimeters';
%legenden(1).l.Position   = [10.2 1.8 0 0];

legenden(1).l.Box        = 'off';



%% Speichern und exportieren
filename = 'template_einfach';
set(gcf,'PaperPositionMode','auto');
print('-depsc',[filename,'_tmp.eps']); %Ausgabe als .eps
system(['gswin32c.exe -dNOPAUSE -dBATCH -dNOCACHE -dEPSCrop -sDEVICE=epswrite -sOutputFile=',filename,'.eps ',filename,'_tmp.eps']);
system(['gswin32c.exe -dNOPAUSE -dBATCH -dNOCACHE -dEPSCrop -sDEVICE=pdfwrite -sOutputFile=',filename,'.pdf ',filename,'_tmp.eps']);
system(['gswin32c.exe -dNOPAUSE -dBATCH -dNOCACHE -r300 -dEPSCrop -sDEVICE=png16m   -sOutputFile=',filename,'.png ',filename,'_tmp.eps']);

