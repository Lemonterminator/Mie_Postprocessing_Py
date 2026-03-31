\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aaltothesis}[2026/03/27 Minimal compatibility class]

\DeclareOption{english}{}
\DeclareOption{finnish}{}
\DeclareOption{swedish}{}
\DeclareOption{utf8}{}
\DeclareOption{latin1}{}
\DeclareOption{a-1b}{}
\DeclareOption{a-2b}{}
\DeclareOption{online}{}
\DeclareOption{print}{}
\DeclareOption{elec}{}
\DeclareOption{arts}{}
\DeclareOption{biz}{}
\DeclareOption{chem}{}
\DeclareOption{eng}{}
\DeclareOption{sci}{}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{article}

\PassOptionsToPackage{hyperfootnotes=false}{hyperref}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{siunitx}
\RequirePackage{titlesec}
\RequirePackage{amsmath}
\RequirePackage{amssymb}

\geometry{margin=1in}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{1.2em}
\hfuzz=999pt
\hbadness=10000
\overfullrule=0pt

\newcommand{\setupthesisfonts}{}
\newcommand{\spc}{; }
\newcommand{\dothesispagenumbering}{\clearpage\pagenumbering{arabic}}
\newcommand{\mysection}[1]{\section*{#1}\addcontentsline{toc}{section}{#1}}
\newcommand{\thesistableofcontents}{\tableofcontents\clearpage}
\newcommand{\thesisbibliography}{\clearpage\printbibliography[heading=bibintoc,title={References}]}
\newcommand{\abstracttext}[1]{#1}

\def\@degreeprogram{}
\def\@major{}
\def\@univdegree{}
\def\@thesisauthor{}
\def\@thesistitle{}
\def\@thesissubtitle{}
\def\@place{}
\def\@supervisor{}
\def\@advisor{}
\def\@keywords{}
\def\@thesisabstract{}
\def\@date{\today}

\newcommand{\degreeprogram}[1]{\gdef\@degreeprogram{#1}}
\newcommand{\major}[1]{\gdef\@major{#1}}
\newcommand{\univdegree}[1]{\gdef\@univdegree{#1}}
\newcommand{\thesisauthor}[1]{\gdef\@thesisauthor{#1}\author{#1}}
\newcommand{\ThesisAuthor}{\@thesisauthor}
\newcommand{\thesistitle}[2][]{\gdef\@thesistitle{#2}\title{#2}}
\newcommand{\thesissubtitle}[2][]{\gdef\@thesissubtitle{#2}}
\renewcommand{\date}[1]{\gdef\@date{#1}}
\newcommand{\place}[1]{\gdef\@place{#1}}
\newcommand{\supervisor}[1]{\gdef\@supervisor{#1}}
\newcommand{\advisor}[1]{\ifx\@advisor\@empty\gdef\@advisor{#1}\else\g@addto@macro\@advisor{\\#1}\fi}
\newcommand{\collaborativepartner}[1]{}
\newcommand{\uselogo}[1]{}
\newcommand{\copyrighttext}[2]{}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}
\newcommand{\thesisabstract}[1]{\gdef\@thesisabstract{#1}}

\newcommand{\makecoverpage}{%
  \begin{titlepage}
  \centering
  {\Large \@degreeprogram\par}
  \vspace{1em}
  {\large \@major\par}
  \vspace{3em}
  {\LARGE\bfseries \@thesistitle\par}
  \ifx\@thesissubtitle\@empty\else\vspace{1em}{\Large \@thesissubtitle\par}\fi
  \vfill
  {\large \@thesisauthor\par}
  \vspace{1em}
  {\@place\par}
  {\@date\par}
  \end{titlepage}
}

\newcommand{\makecopyrightpage}{\clearpage\thispagestyle{empty}\null\clearpage}

\newenvironment{abstractpage}[1][]
{\section*{Abstract}\noindent\textbf{Keywords:} \@keywords\par\medskip}
{\par\clearpage}

\newcommand{\tiltangle}{\varphi}

\AtBeginDocument{\pagestyle{plain}\pagenumbering{roman}\hypersetup{pageanchor=false}}
