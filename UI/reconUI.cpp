///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun 17 2015)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "reconUI.h"

///////////////////////////////////////////////////////////////////////////

mainWindow::mainWindow( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	this->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	m_mgr.SetManagedWindow(this);
	m_mgr.SetFlags(wxAUI_MGR_DEFAULT);
	
	m_menubar1 = new wxMenuBar( 0|wxNO_BORDER );
	m_menubar1->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_WINDOW ) );
	m_menubar1->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	
	file = new wxMenu();
	wxMenuItem* newPage;
	newPage = new wxMenuItem( file, wxID_NEW, wxString( wxT("New\tCtrl+N") ) , wxEmptyString, wxITEM_NORMAL );
	file->Append( newPage );
	
	wxMenuItem* open;
	open = new wxMenuItem( file, wxID_OPEN, wxString( wxT("Open\tCtrl+O") ) , wxEmptyString, wxITEM_NORMAL );
	file->Append( open );
	
	wxMenuItem* save;
	save = new wxMenuItem( file, wxID_ANY, wxString( wxT("Save") ) + wxT('\t') + wxT("Ctrl+S"), wxEmptyString, wxITEM_NORMAL );
	file->Append( save );
	
	wxMenuItem* quit;
	quit = new wxMenuItem( file, wxID_ANY, wxString( wxT("Exit\tAlt-X") ) , wxEmptyString, wxITEM_NORMAL );
	file->Append( quit );
	
	m_menubar1->Append( file, wxT("File") ); 
	
	config = new wxMenu();
	wxMenuItem* configDialog;
	configDialog = new wxMenuItem( config, wxID_PREFERENCES, wxString( wxT("Settings") ) , wxEmptyString, wxITEM_NORMAL );
	config->Append( configDialog );
	
	wxMenuItem* gainSelect;
	gainSelect = new wxMenuItem( config, wxID_ANY, wxString( wxT("Edit Gain Files") ) , wxEmptyString, wxITEM_NORMAL );
	config->Append( gainSelect );
	
	m_menubar1->Append( config, wxT("Config") ); 
	
	calibration = new wxMenu();
	wxMenuItem* resList;
	resList = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Set Resolution Phantoms") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( resList );
	
	wxMenuItem* contList;
	contList = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Set Contrast Phantom") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( contList );
	
	wxMenuItem* runTest;
	runTest = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Run Tests") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( runTest );
	
	wxMenuItem* testGeo;
	testGeo = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Test Geometries") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( testGeo );
	
	wxMenuItem* autoGeo;
	autoGeo = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Auto-detect Geometry") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( autoGeo );
	
	m_menubar1->Append( calibration, wxT("Calibration") ); 
	
	help = new wxMenu();
	wxMenuItem* about;
	about = new wxMenuItem( help, wxID_ABOUT, wxString( wxT("About\tF1") ) , wxEmptyString, wxITEM_NORMAL );
	help->Append( about );
	
	m_menubar1->Append( help, wxT("Help") ); 
	
	this->SetMenuBar( m_menubar1 );
	
	wxString optionBoxChoices[] = { wxT("Navigation"), wxT("Edge Enhancement"), wxT("Scan Line Removal"), wxT("Denoising") };
	int optionBoxNChoices = sizeof( optionBoxChoices ) / sizeof( wxString );
	optionBox = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, optionBoxNChoices, optionBoxChoices, 0 );
	optionBox->SetSelection( 0 );
	m_mgr.AddPane( optionBox, wxAuiPaneInfo() .Top() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Movable( false ).Dock().Fixed().BottomDockable( false ).LeftDockable( false ).RightDockable( false ).Floatable( false ).Layer( 10 ) );
	
	navToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER ); 
	navToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	navToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	
	distanceLabel = new wxStaticText( navToolbar, wxID_ANY, wxT("Current Distance"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceLabel->Wrap( -1 );
	navToolbar->AddControl( distanceLabel );
	distanceValue = new wxTextCtrl( navToolbar, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxSize( 50,-1 ), wxTE_PROCESS_ENTER );
	navToolbar->AddControl( distanceValue );
	distanceUnits = new wxStaticText( navToolbar, wxID_ANY, wxT("mm"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceUnits->Wrap( -1 );
	navToolbar->AddControl( distanceUnits );
	autoFocus = new wxButton( navToolbar, wxID_ANY, wxT("Auto-focus"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( autoFocus );
	stepLabel = new wxStaticText( navToolbar, wxID_ANY, wxT("Step size:"), wxDefaultPosition, wxDefaultSize, 0 );
	stepLabel->Wrap( -1 );
	navToolbar->AddControl( stepLabel );
	stepVal = new wxStaticText( navToolbar, wxID_ANY, wxT("0.5"), wxDefaultPosition, wxDefaultSize, 0 );
	stepVal->Wrap( -1 );
	navToolbar->AddControl( stepVal );
	stepUnits = new wxStaticText( navToolbar, wxID_ANY, wxT("mm"), wxDefaultPosition, wxDefaultSize, 0 );
	stepUnits->Wrap( -1 );
	navToolbar->AddControl( stepUnits );
	stepSlider = new wxSlider( navToolbar, wxID_ANY, 5, 1, 10, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	navToolbar->AddControl( stepSlider );
	autoLight = new wxButton( navToolbar, wxID_ANY, wxT("Auto W+L"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( autoLight );
	windowLabel = new wxStaticText( navToolbar, wxID_ANY, wxT("Window:"), wxDefaultPosition, wxDefaultSize, 0 );
	windowLabel->Wrap( -1 );
	navToolbar->AddControl( windowLabel );
	windowVal = new wxStaticText( navToolbar, wxID_ANY, wxT("65535"), wxDefaultPosition, wxDefaultSize, 0 );
	windowVal->Wrap( -1 );
	navToolbar->AddControl( windowVal );
	windowSlider = new wxSlider( navToolbar, wxID_ANY, 255, 1, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	navToolbar->AddControl( windowSlider );
	levelLabel = new wxStaticText( navToolbar, wxID_ANY, wxT("Level:"), wxDefaultPosition, wxDefaultSize, 0 );
	levelLabel->Wrap( -1 );
	navToolbar->AddControl( levelLabel );
	levelVal = new wxStaticText( navToolbar, wxID_ANY, wxT("10000"), wxDefaultPosition, wxDefaultSize, 0 );
	levelVal->Wrap( -1 );
	navToolbar->AddControl( levelVal );
	levelSlider = new wxSlider( navToolbar, wxID_ANY, 39, 0, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	navToolbar->AddControl( levelSlider );
	zoomLabel = new wxStaticText( navToolbar, wxID_ANY, wxT("Zoom:"), wxDefaultPosition, wxDefaultSize, 0 );
	zoomLabel->Wrap( -1 );
	navToolbar->AddControl( zoomLabel );
	zoomVal = new wxStaticText( navToolbar, wxID_ANY, wxT("1.0"), wxDefaultPosition, wxDefaultSize, 0 );
	zoomVal->Wrap( -1 );
	navToolbar->AddControl( zoomVal );
	zoomUnits = new wxStaticText( navToolbar, wxID_ANY, wxT("x"), wxDefaultPosition, wxDefaultSize, 0 );
	zoomUnits->Wrap( -1 );
	navToolbar->AddControl( zoomUnits );
	zoomSlider = new wxSlider( navToolbar, wxID_ANY, 0, 0, 20, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	navToolbar->AddControl( zoomSlider );
	autoAll = new wxButton( navToolbar, wxID_ANY, wxT("Auto Focus and Light"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( autoAll );
	vertFlip = new wxCheckBox( navToolbar, wxID_ANY, wxT("Flip Vertical"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( vertFlip );
	horFlip = new wxCheckBox( navToolbar, wxID_ANY, wxT("Flip Horizontal"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( horFlip );
	logView = new wxCheckBox( navToolbar, wxID_ANY, wxT("Log view"), wxDefaultPosition, wxDefaultSize, 0 );
	logView->SetValue(true); 
	navToolbar->AddControl( logView );
	projectionView = new wxCheckBox( navToolbar, wxID_ANY, wxT("View projections"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( projectionView );
	navToolbar->Realize();
	m_mgr.AddPane( navToolbar, wxAuiPaneInfo() .Top() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Movable( false ).Dock().Resizable().FloatingSize( wxDefaultSize ).DockFixed( true ).BottomDockable( false ).TopDockable( false ).LeftDockable( false ).RightDockable( false ).Floatable( false ).Layer( 10 ) );
	
	
	edgeToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER ); 
	edgeToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	edgeToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	
	xEnhance = new wxCheckBox( edgeToolbar, wxID_ANY, wxT("X"), wxDefaultPosition, wxDefaultSize, 0 );
	xEnhance->SetValue(true); 
	edgeToolbar->AddControl( xEnhance );
	yEnhance = new wxCheckBox( edgeToolbar, wxID_ANY, wxT("Y"), wxDefaultPosition, wxDefaultSize, 0 );
	yEnhance->SetValue(true); 
	edgeToolbar->AddControl( yEnhance );
	absEnhance = new wxCheckBox( edgeToolbar, wxID_ANY, wxT("Abs"), wxDefaultPosition, wxDefaultSize, 0 );
	absEnhance->SetValue(true); 
	edgeToolbar->AddControl( absEnhance );
	ratioLabel = new wxStaticText( edgeToolbar, wxID_ANY, wxT("Edge/Image ratio: "), wxDefaultPosition, wxDefaultSize, 0 );
	ratioLabel->Wrap( -1 );
	ratioLabel->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	
	edgeToolbar->AddControl( ratioLabel );
	ratioValue = new wxStaticText( edgeToolbar, wxID_ANY, wxT("5.0"), wxDefaultPosition, wxDefaultSize, 0 );
	ratioValue->Wrap( -1 );
	edgeToolbar->AddControl( ratioValue );
	resetEnhance = new wxButton( edgeToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	edgeToolbar->AddControl( resetEnhance );
	enhanceSlider = new wxSlider( edgeToolbar, wxID_ANY, 50, 0, 200, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	edgeToolbar->AddControl( enhanceSlider );
	edgeToolbar->Realize();
	m_mgr.AddPane( edgeToolbar, wxAuiPaneInfo() .Top() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Movable( false ).Dock().Resizable().FloatingSize( wxDefaultSize ).DockFixed( true ).BottomDockable( false ).TopDockable( false ).LeftDockable( false ).RightDockable( false ).Floatable( false ).Layer( 10 ) );
	
	
	scanToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER|wxNO_BORDER ); 
	scanToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	scanToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	
	scanVertEnable = new wxCheckBox( scanToolbar, wxID_ANY, wxT("Scanline vertical correction factor: "), wxDefaultPosition, wxDefaultSize, 0 );
	scanVertEnable->SetValue(true); 
	scanToolbar->AddControl( scanVertEnable );
	scanVertValue = new wxStaticText( scanToolbar, wxID_ANY, wxT("0.25"), wxDefaultPosition, wxDefaultSize, 0 );
	scanVertValue->Wrap( -1 );
	scanVertValue->Hide();
	
	scanToolbar->AddControl( scanVertValue );
	resetScanVert = new wxButton( scanToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	scanToolbar->AddControl( resetScanVert );
	scanVertSlider = new wxSlider( scanToolbar, wxID_ANY, 25, 0, 50, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	scanVertSlider->Hide();
	
	scanToolbar->AddControl( scanVertSlider );
	scanHorEnable = new wxCheckBox( scanToolbar, wxID_ANY, wxT("Scanline horizontal correction factor: "), wxDefaultPosition, wxDefaultSize, 0 );
	scanToolbar->AddControl( scanHorEnable );
	scanHorValue = new wxStaticText( scanToolbar, wxID_ANY, wxT("0.1"), wxDefaultPosition, wxDefaultSize, 0 );
	scanHorValue->Wrap( -1 );
	scanHorValue->Enable( false );
	scanHorValue->Hide();
	
	scanToolbar->AddControl( scanHorValue );
	resetScanHor = new wxButton( scanToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	resetScanHor->Enable( false );
	
	scanToolbar->AddControl( resetScanHor );
	scanHorSlider = new wxSlider( scanToolbar, wxID_ANY, 10, 0, 50, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	scanHorSlider->Enable( false );
	scanHorSlider->Hide();
	
	scanToolbar->AddControl( scanHorSlider );
	scanToolbar->Realize();
	m_mgr.AddPane( scanToolbar, wxAuiPaneInfo() .Top() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Movable( false ).Dock().Resizable().FloatingSize( wxDefaultSize ).DockFixed( true ).BottomDockable( false ).TopDockable( false ).LeftDockable( false ).RightDockable( false ).Floatable( false ).Layer( 10 ) );
	
	
	noiseToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER ); 
	noiseToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	noiseToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	
	outlierEnable = new wxCheckBox( noiseToolbar, wxID_ANY, wxT("Large noise hard removal over value:  "), wxDefaultPosition, wxDefaultSize, 0 );
	outlierEnable->SetValue(true); 
	noiseToolbar->AddControl( outlierEnable );
	noiseMaxVal = new wxStaticText( noiseToolbar, wxID_ANY, wxT("30"), wxDefaultPosition, wxDefaultSize, 0 );
	noiseMaxVal->Wrap( -1 );
	noiseToolbar->AddControl( noiseMaxVal );
	resetNoiseMax = new wxButton( noiseToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	noiseToolbar->AddControl( resetNoiseMax );
	noiseMaxSlider = new wxSlider( noiseToolbar, wxID_ANY, 30, 10, 50, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	noiseToolbar->AddControl( noiseMaxSlider );
	noiseToolbar->Realize();
	m_mgr.AddPane( noiseToolbar, wxAuiPaneInfo() .Top() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Movable( false ).Dock().Resizable().FloatingSize( wxDefaultSize ).DockFixed( true ).BottomDockable( false ).TopDockable( false ).LeftDockable( false ).RightDockable( false ).Floatable( false ).Layer( 10 ) );
	
	
	m_auinotebook6 = new wxAuiNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0|wxNO_BORDER );
	m_mgr.AddPane( m_auinotebook6, wxAuiPaneInfo() .Left() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Dock().Resizable().FloatingSize( wxDefaultSize ).CentrePane() );
	
	m_panel10 = new wxPanel( m_auinotebook6, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_panel10->SetToolTip( wxT("Console") );
	
	wxBoxSizer* bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL );
	
	m_textCtrl8 = new wxTextCtrl( m_panel10, wxID_ANY, wxT("If this is your first run, make sure you set calibration and config files in the \"Config\" menu above. Any settings you change will be automatically saved for future sessions on this machine.\nOpen a series of 7 projections in the same folder using \"New\".\nReconstructions can be done either with \"enter\" or \"f5\" keys. Interactive mode (f5) will allow you to see the reconstruction as it is running, but takes significantly longer than normal mode (enter).\nLog and error outputs from the reconstructions on any tab will be displayed below this text.\n"), wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer1->Add( m_textCtrl8, 1, wxALL|wxEXPAND, 5 );
	
	
	m_panel10->SetSizer( bSizer1 );
	m_panel10->Layout();
	bSizer1->Fit( m_panel10 );
	m_auinotebook6->AddPage( m_panel10, wxT("Console"), false, wxNullBitmap );
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	
	m_mgr.Update();
	this->Centre( wxBOTH );
	
	// Connect Events
	this->Connect( wxEVT_KEY_DOWN, wxKeyEventHandler( mainWindow::onKeyDown ) );
	this->Connect( wxEVT_KEY_UP, wxKeyEventHandler( mainWindow::onKeyUp ) );
	this->Connect( newPage->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onNew ) );
	this->Connect( open->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onOpen ) );
	this->Connect( save->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onSave ) );
	this->Connect( quit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onQuit ) );
	this->Connect( configDialog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onConfig ) );
	this->Connect( gainSelect->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onGainSelect ) );
	this->Connect( resList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onResList ) );
	this->Connect( contList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onContList ) );
	this->Connect( runTest->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onRunTest ) );
	this->Connect( testGeo->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onTestGeo ) );
	this->Connect( autoGeo->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAutoGeo ) );
	this->Connect( about->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAbout ) );
	optionBox->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( mainWindow::onToolbarChoice ), NULL, this );
	distanceValue->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( mainWindow::onDistance ), NULL, this );
	autoFocus->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoFocus ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	autoLight->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoLight ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	autoAll->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoAll ), NULL, this );
	vertFlip->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onVertFlip ), NULL, this );
	horFlip->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onHorFlip ), NULL, this );
	logView->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onLogView ), NULL, this );
	projectionView->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onProjectionView ), NULL, this );
	xEnhance->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onXEnhance ), NULL, this );
	yEnhance->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onYEnhance ), NULL, this );
	absEnhance->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onAbsEnhance ), NULL, this );
	resetEnhance->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetEnhance ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	scanVertEnable->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onScanVertEnable ), NULL, this );
	resetScanVert->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanHorEnable->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onScanHorEnable ), NULL, this );
	resetScanHor->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	outlierEnable->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onNoiseMaxEnable ), NULL, this );
	resetNoiseMax->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	m_auinotebook6->Connect( wxEVT_COMMAND_AUINOTEBOOK_PAGE_CHANGING, wxAuiNotebookEventHandler( mainWindow::onPageChange ), NULL, this );
	m_auinotebook6->Connect( wxEVT_COMMAND_AUINOTEBOOK_PAGE_CLOSE, wxAuiNotebookEventHandler( mainWindow::onPageClose ), NULL, this );
}

mainWindow::~mainWindow()
{
	// Disconnect Events
	this->Disconnect( wxEVT_KEY_DOWN, wxKeyEventHandler( mainWindow::onKeyDown ) );
	this->Disconnect( wxEVT_KEY_UP, wxKeyEventHandler( mainWindow::onKeyUp ) );
	this->Disconnect( wxID_NEW, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onNew ) );
	this->Disconnect( wxID_OPEN, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onOpen ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onSave ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onQuit ) );
	this->Disconnect( wxID_PREFERENCES, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onConfig ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onGainSelect ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onResList ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onContList ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onRunTest ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onTestGeo ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAutoGeo ) );
	this->Disconnect( wxID_ABOUT, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAbout ) );
	optionBox->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( mainWindow::onToolbarChoice ), NULL, this );
	distanceValue->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( mainWindow::onDistance ), NULL, this );
	autoFocus->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoFocus ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onStepSlider ), NULL, this );
	autoLight->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoLight ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	windowSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onWindowSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	levelSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onLevelSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	zoomSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onZoomSlider ), NULL, this );
	autoAll->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoAll ), NULL, this );
	vertFlip->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onVertFlip ), NULL, this );
	horFlip->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onHorFlip ), NULL, this );
	logView->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onLogView ), NULL, this );
	projectionView->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onProjectionView ), NULL, this );
	xEnhance->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onXEnhance ), NULL, this );
	yEnhance->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onYEnhance ), NULL, this );
	absEnhance->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onAbsEnhance ), NULL, this );
	resetEnhance->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetEnhance ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	enhanceSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onEnhanceRatio ), NULL, this );
	scanVertEnable->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onScanVertEnable ), NULL, this );
	resetScanVert->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onScanVert ), NULL, this );
	scanHorEnable->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onScanHorEnable ), NULL, this );
	resetScanHor->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onScanHor ), NULL, this );
	outlierEnable->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( mainWindow::onNoiseMaxEnable ), NULL, this );
	resetNoiseMax->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onResetNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( mainWindow::onNoiseMax ), NULL, this );
	m_auinotebook6->Disconnect( wxEVT_COMMAND_AUINOTEBOOK_PAGE_CHANGING, wxAuiNotebookEventHandler( mainWindow::onPageChange ), NULL, this );
	m_auinotebook6->Disconnect( wxEVT_COMMAND_AUINOTEBOOK_PAGE_CLOSE, wxAuiNotebookEventHandler( mainWindow::onPageClose ), NULL, this );
	
	m_mgr.UnInit();
	
}

RunBox::RunBox( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_gauge2 = new wxGauge( this, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	m_gauge2->SetValue( 0 ); 
	fgSizer2->Add( m_gauge2, 0, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	this->SetSizer( fgSizer2 );
	this->Layout();
	
	this->Centre( wxBOTH );
}

RunBox::~RunBox()
{
}

resDialog::resDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxFlexGridSizer* fgSizer3;
	fgSizer3 = new wxFlexGridSizer( 4, 0, 0, 0 );
	fgSizer3->SetFlexibleDirection( wxVERTICAL );
	fgSizer3->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_listCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxSize( 1050,300 ), wxLC_REPORT );
	fgSizer3->Add( m_listCtrl, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	wxGridSizer* gSizer2;
	gSizer2 = new wxGridSizer( 0, 2, 0, 0 );
	
	addNew = new wxButton( this, wxID_ANY, wxT("Add New Image Sets"), wxPoint( 10,-1 ), wxDefaultSize, 0 );
	gSizer2->Add( addNew, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	remove = new wxButton( this, wxID_ANY, wxT("Remove Selected"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer2->Add( remove, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	fgSizer3->Add( gSizer2, 1, wxEXPAND, 5 );
	
	m_staticline3 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	fgSizer3->Add( m_staticline3, 0, wxEXPAND | wxALL, 5 );
	
	wxGridSizer* gSizer3;
	gSizer3 = new wxGridSizer( 0, 2, 0, 0 );
	
	ok = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer3->Add( ok, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	cancel = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer3->Add( cancel, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	fgSizer3->Add( gSizer3, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( fgSizer3 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	addNew->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onAddNew ), NULL, this );
	remove->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onRemove ), NULL, this );
	ok->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onOk ), NULL, this );
	cancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onCancel ), NULL, this );
}

resDialog::~resDialog()
{
	// Disconnect Events
	addNew->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onAddNew ), NULL, this );
	remove->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onRemove ), NULL, this );
	ok->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onOk ), NULL, this );
	cancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( resDialog::onCancel ), NULL, this );
	
}

MyDialog3::MyDialog3( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxGridBagSizer* gbSizer1;
	gbSizer1 = new wxGridBagSizer( 0, 0 );
	gbSizer1->SetFlexibleDirection( wxBOTH );
	gbSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	
	gbSizer1->Add( 0, 0, wxGBPosition( 0, 3 ), wxGBSpan( 1, 1 ), wxEXPAND, 5 );
	
	
	this->SetSizer( gbSizer1 );
	this->Layout();
	
	this->Centre( wxBOTH );
}

MyDialog3::~MyDialog3()
{
}

configDialog::configDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	this->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_WINDOW ) );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 0, 4, 0, 0 );
	fgSizer2->SetFlexibleDirection( wxVERTICAL );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText13 = new wxStaticText( this, wxID_ANY, wxT("Detector information"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText13->Wrap( -1 );
	fgSizer2->Add( m_staticText13, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxGridSizer* gSizer2;
	gSizer2 = new wxGridSizer( 0, 2, 0, 0 );
	
	m_staticText9 = new wxStaticText( this, wxID_ANY, wxT("Height (pixels)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText9->Wrap( -1 );
	gSizer2->Add( m_staticText9, 0, wxALL, 5 );
	
	pixelWidth = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	gSizer2->Add( pixelWidth, 0, wxALL, 5 );
	
	m_staticText10 = new wxStaticText( this, wxID_ANY, wxT("Width (pixels)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText10->Wrap( -1 );
	gSizer2->Add( m_staticText10, 0, wxALL, 5 );
	
	pixelHeight = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	gSizer2->Add( pixelHeight, 0, wxALL, 5 );
	
	m_staticText11 = new wxStaticText( this, wxID_ANY, wxT("Pitch height (mm)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText11->Wrap( -1 );
	gSizer2->Add( m_staticText11, 0, wxALL, 5 );
	
	pitchHeight = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	gSizer2->Add( pitchHeight, 0, wxALL, 5 );
	
	m_staticText12 = new wxStaticText( this, wxID_ANY, wxT("Pitch width (mm)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText12->Wrap( -1 );
	gSizer2->Add( m_staticText12, 0, wxALL, 5 );
	
	pitchWidth = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	gSizer2->Add( pitchWidth, 0, wxALL, 5 );
	
	
	fgSizer2->Add( gSizer2, 1, wxEXPAND, 5 );
	
	m_staticText14 = new wxStaticText( this, wxID_ANY, wxT("Beam emitter locations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText14->Wrap( -1 );
	fgSizer2->Add( m_staticText14, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_grid1 = new wxGrid( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	
	// Grid
	m_grid1->CreateGrid( 7, 3 );
	m_grid1->EnableEditing( true );
	m_grid1->EnableGridLines( true );
	m_grid1->EnableDragGridSize( false );
	m_grid1->SetMargins( 0, 0 );
	
	// Columns
	m_grid1->EnableDragColMove( false );
	m_grid1->EnableDragColSize( true );
	m_grid1->SetColLabelSize( 30 );
	m_grid1->SetColLabelValue( 0, wxT("x (mm)") );
	m_grid1->SetColLabelValue( 1, wxT("y (mm)") );
	m_grid1->SetColLabelValue( 2, wxT("z (mm)") );
	m_grid1->SetColLabelAlignment( wxALIGN_CENTRE, wxALIGN_CENTRE );
	
	// Rows
	m_grid1->EnableDragRowSize( true );
	m_grid1->SetRowLabelSize( 80 );
	m_grid1->SetRowLabelAlignment( wxALIGN_CENTRE, wxALIGN_CENTRE );
	
	// Label Appearance
	
	// Cell Defaults
	m_grid1->SetDefaultCellAlignment( wxALIGN_LEFT, wxALIGN_TOP );
	fgSizer2->Add( m_grid1, 0, wxALL|wxEXPAND, 5 );
	
	loadConfig = new wxButton( this, wxID_ANY, wxT("Load Config"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( loadConfig, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	saveConfig = new wxButton( this, wxID_ANY, wxT("Save Config"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( saveConfig, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	ok = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( ok, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	cancel = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( cancel, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	this->SetSizer( fgSizer2 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	pixelWidth->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	pixelHeight->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	pitchHeight->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	pitchWidth->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	loadConfig->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onLoad ), NULL, this );
	saveConfig->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onSave ), NULL, this );
	ok->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onOK ), NULL, this );
	cancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onCancel ), NULL, this );
}

configDialog::~configDialog()
{
	// Disconnect Events
	pixelWidth->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	pixelHeight->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	pitchHeight->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	pitchWidth->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( configDialog::onConfigChar ), NULL, this );
	loadConfig->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onLoad ), NULL, this );
	saveConfig->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onSave ), NULL, this );
	ok->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onOK ), NULL, this );
	cancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onCancel ), NULL, this );
	
}
