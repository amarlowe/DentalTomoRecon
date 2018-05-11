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
	this->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_APPWORKSPACE ) );
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
	save->Enable( false );
	
	wxMenuItem* exportRecon;
	exportRecon = new wxMenuItem( file, wxID_ANY, wxString( wxT("Export Reconstruction") ) + wxT('\t') + wxT("Ctrl+E"), wxEmptyString, wxITEM_NORMAL );
	file->Append( exportRecon );
	exportRecon->Enable( false );
	
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
	
	wxMenuItem* reconSetup;
	reconSetup = new wxMenuItem( config, wxID_ANY, wxString( wxT("Edit Reconstruction Settings") ) + wxT('\t') + wxT("F5"), wxEmptyString, wxITEM_NORMAL );
	config->Append( reconSetup );
	reconSetup->Enable( false );
	
	m_menubar1->Append( config, wxT("Config") ); 
	
	calibration = new wxMenu();
	wxMenuItem* resList;
	resList = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Set Resolution Phantoms") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( resList );
	
	wxMenuItem* contList;
	contList = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Set Contrast Phantom") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( contList );
	contList->Enable( false );
	
	wxMenuItem* runTest;
	runTest = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Run Tests") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( runTest );
	runTest->Enable( false );
	
	wxMenuItem* testGeo;
	testGeo = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Test Geometries") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( testGeo );
	
	wxMenuItem* autoGeo;
	autoGeo = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Auto-detect Geometry (Bead)") ) , wxEmptyString, wxITEM_NORMAL );
	calibration->Append( autoGeo );
	
	wxMenuItem* autoGeoS;
	autoGeoS = new wxMenuItem( calibration, wxID_ANY, wxString( wxT("Auto-detect Geometry (Selection)") ) + wxT('\t') + wxT("Ctrl+G"), wxEmptyString, wxITEM_NORMAL );
	calibration->Append( autoGeoS );
	autoGeoS->Enable( false );
	
	m_menubar1->Append( calibration, wxT("Calibration") ); 
	
	help = new wxMenu();
	wxMenuItem* about;
	about = new wxMenuItem( help, wxID_ABOUT, wxString( wxT("About\tF1") ) , wxEmptyString, wxITEM_NORMAL );
	help->Append( about );
	
	m_menubar1->Append( help, wxT("Help") ); 
	
	this->SetMenuBar( m_menubar1 );
	
	wxString optionBoxChoices[] = { wxT("Navigation"), wxT("Edge Enhancement") };
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
	distanceValue->Enable( false );
	
	navToolbar->AddControl( distanceValue );
	distanceUnits = new wxStaticText( navToolbar, wxID_ANY, wxT("mm"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceUnits->Wrap( -1 );
	navToolbar->AddControl( distanceUnits );
	autoFocus = new wxButton( navToolbar, wxID_ANY, wxT("Auto-focus"), wxDefaultPosition, wxDefaultSize, 0 );
	autoFocus->Enable( false );
	
	navToolbar->AddControl( autoFocus );
	autoLight = new wxButton( navToolbar, wxID_ANY, wxT("Auto W+L"), wxDefaultPosition, wxDefaultSize, 0 );
	autoLight->Enable( false );
	
	navToolbar->AddControl( autoLight );
	windowLabel = new wxStaticText( navToolbar, wxID_ANY, wxT("Window:"), wxDefaultPosition, wxDefaultSize, 0 );
	windowLabel->Wrap( -1 );
	navToolbar->AddControl( windowLabel );
	windowVal = new wxStaticText( navToolbar, wxID_ANY, wxT("65535"), wxDefaultPosition, wxDefaultSize, 0 );
	windowVal->Wrap( -1 );
	navToolbar->AddControl( windowVal );
	windowSlider = new wxSlider( navToolbar, wxID_ANY, 255, 1, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	windowSlider->Enable( false );
	
	navToolbar->AddControl( windowSlider );
	levelLabel = new wxStaticText( navToolbar, wxID_ANY, wxT("Level:"), wxDefaultPosition, wxDefaultSize, 0 );
	levelLabel->Wrap( -1 );
	navToolbar->AddControl( levelLabel );
	levelVal = new wxStaticText( navToolbar, wxID_ANY, wxT("10000"), wxDefaultPosition, wxDefaultSize, 0 );
	levelVal->Wrap( -1 );
	navToolbar->AddControl( levelVal );
	levelSlider = new wxSlider( navToolbar, wxID_ANY, 39, 0, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	levelSlider->Enable( false );
	
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
	zoomSlider->Enable( false );
	
	navToolbar->AddControl( zoomSlider );
	autoAll = new wxButton( navToolbar, wxID_ANY, wxT("Auto Focus and Light"), wxDefaultPosition, wxDefaultSize, 0 );
	autoAll->Enable( false );
	
	navToolbar->AddControl( autoAll );
	vertFlip = new wxCheckBox( navToolbar, wxID_ANY, wxT("Flip Vertical"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( vertFlip );
	horFlip = new wxCheckBox( navToolbar, wxID_ANY, wxT("Flip Horizontal"), wxDefaultPosition, wxDefaultSize, 0 );
	navToolbar->AddControl( horFlip );
	logView = new wxCheckBox( navToolbar, wxID_ANY, wxT("Log view"), wxDefaultPosition, wxDefaultSize, 0 );
	logView->SetValue(true); 
	navToolbar->AddControl( logView );
	wxString dataDisplayChoices[] = { wxT("Reconstruction"), wxT("Single Pass Reconstruction"), wxT("Projections"), wxT("Synthetic 2D"), wxT("Error") };
	int dataDisplayNChoices = sizeof( dataDisplayChoices ) / sizeof( wxString );
	dataDisplay = new wxChoice( navToolbar, wxID_ANY, wxDefaultPosition, wxDefaultSize, dataDisplayNChoices, dataDisplayChoices, 0 );
	dataDisplay->SetSelection( 0 );
	navToolbar->AddControl( dataDisplay );
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
	ratioValue = new wxStaticText( edgeToolbar, wxID_ANY, wxT("0.5"), wxDefaultPosition, wxDefaultSize, 0 );
	ratioValue->Wrap( -1 );
	edgeToolbar->AddControl( ratioValue );
	resetEnhance = new wxButton( edgeToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	edgeToolbar->AddControl( resetEnhance );
	enhanceSlider = new wxSlider( edgeToolbar, wxID_ANY, 50, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	edgeToolbar->AddControl( enhanceSlider );
	edgeToolbar->Realize();
	m_mgr.AddPane( edgeToolbar, wxAuiPaneInfo() .Top() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Movable( false ).Dock().Resizable().FloatingSize( wxDefaultSize ).DockFixed( true ).BottomDockable( false ).TopDockable( false ).LeftDockable( false ).RightDockable( false ).Floatable( false ).Layer( 10 ) );
	
	
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
	this->Connect( exportRecon->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onExportRecon ) );
	this->Connect( quit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onQuit ) );
	this->Connect( configDialog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onConfig ) );
	this->Connect( gainSelect->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onGainSelect ) );
	this->Connect( reconSetup->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onReconSetup ) );
	this->Connect( resList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onResList ) );
	this->Connect( contList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onContList ) );
	this->Connect( runTest->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onRunTest ) );
	this->Connect( testGeo->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onTestGeo ) );
	this->Connect( autoGeo->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAutoGeo ) );
	this->Connect( autoGeoS->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAutoGeoS ) );
	this->Connect( about->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAbout ) );
	optionBox->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( mainWindow::onToolbarChoice ), NULL, this );
	distanceValue->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( mainWindow::onDistance ), NULL, this );
	autoFocus->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoFocus ), NULL, this );
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
	dataDisplay->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( mainWindow::onDataDisplay ), NULL, this );
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
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onExportRecon ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onQuit ) );
	this->Disconnect( wxID_PREFERENCES, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onConfig ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onGainSelect ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onReconSetup ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onResList ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onContList ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onRunTest ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onTestGeo ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAutoGeo ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAutoGeoS ) );
	this->Disconnect( wxID_ABOUT, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAbout ) );
	optionBox->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( mainWindow::onToolbarChoice ), NULL, this );
	distanceValue->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( mainWindow::onDistance ), NULL, this );
	autoFocus->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( mainWindow::onAutoFocus ), NULL, this );
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
	dataDisplay->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( mainWindow::onDataDisplay ), NULL, this );
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

reconConfig::reconConfig( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	bSizer6 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer5;
	bSizer5 = new wxBoxSizer( wxHORIZONTAL );
	
	wxString optionBoxChoices[] = { wxT("Distance Selection"), wxT("Exposure"), wxT("Metal Threshold"), wxT("Scan Line Removal"), wxT("Denoising") };
	int optionBoxNChoices = sizeof( optionBoxChoices ) / sizeof( wxString );
	optionBox = new wxChoice( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, optionBoxNChoices, optionBoxChoices, 0 );
	optionBox->SetSelection( 0 );
	bSizer5->Add( optionBox, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	distanceToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER|wxNO_BORDER ); 
	distanceLabel = new wxStaticText( distanceToolbar, wxID_ANY, wxT("Currently displayed distance: "), wxDefaultPosition, wxDefaultSize, 0 );
	distanceLabel->Wrap( -1 );
	distanceToolbar->AddControl( distanceLabel );
	distance = new wxTextCtrl( distanceToolbar, wxID_ANY, wxT("5.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	distanceToolbar->AddControl( distance );
	distanceUnits = new wxStaticText( distanceToolbar, wxID_ANY, wxT("mm  "), wxDefaultPosition, wxDefaultSize, 0 );
	distanceUnits->Wrap( -1 );
	distanceToolbar->AddControl( distanceUnits );
	setStartDis = new wxButton( distanceToolbar, wxID_ANY, wxT("Set As Start Distance"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceToolbar->AddControl( setStartDis );
	setEndDis = new wxButton( distanceToolbar, wxID_ANY, wxT("Set As End Distance"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceToolbar->AddControl( setEndDis );
	m_staticline2 = new wxStaticLine( distanceToolbar, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	distanceToolbar->AddControl( m_staticline2 );
	startDistanceLabel = new wxStaticText( distanceToolbar, wxID_ANY, wxT("Start Distance:"), wxDefaultPosition, wxDefaultSize, 0 );
	startDistanceLabel->Wrap( -1 );
	distanceToolbar->AddControl( startDistanceLabel );
	startDistance = new wxTextCtrl( distanceToolbar, wxID_ANY, wxT("0.0"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceToolbar->AddControl( startDistance );
	startDistanceUnits = new wxStaticText( distanceToolbar, wxID_ANY, wxT("mm"), wxDefaultPosition, wxDefaultSize, 0 );
	startDistanceUnits->Wrap( -1 );
	distanceToolbar->AddControl( startDistanceUnits );
	m_staticline3 = new wxStaticLine( distanceToolbar, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	distanceToolbar->AddControl( m_staticline3 );
	endDistanceLabel = new wxStaticText( distanceToolbar, wxID_ANY, wxT("End Distance:"), wxDefaultPosition, wxDefaultSize, 0 );
	endDistanceLabel->Wrap( -1 );
	distanceToolbar->AddControl( endDistanceLabel );
	endDistance = new wxTextCtrl( distanceToolbar, wxID_ANY, wxT("10.0"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceToolbar->AddControl( endDistance );
	endDistanceUnit = new wxStaticText( distanceToolbar, wxID_ANY, wxT("mm"), wxDefaultPosition, wxDefaultSize, 0 );
	endDistanceUnit->Wrap( -1 );
	distanceToolbar->AddControl( endDistanceUnit );
	stepLabel = new wxStaticText( distanceToolbar, wxID_ANY, wxT("Step size:"), wxDefaultPosition, wxDefaultSize, 0 );
	stepLabel->Wrap( -1 );
	distanceToolbar->AddControl( stepLabel );
	stepVal = new wxStaticText( distanceToolbar, wxID_ANY, wxT("0.5"), wxDefaultPosition, wxDefaultSize, 0 );
	stepVal->Wrap( -1 );
	distanceToolbar->AddControl( stepVal );
	stepUnits = new wxStaticText( distanceToolbar, wxID_ANY, wxT("mm"), wxDefaultPosition, wxDefaultSize, 0 );
	stepUnits->Wrap( -1 );
	distanceToolbar->AddControl( stepUnits );
	stepSlider = new wxSlider( distanceToolbar, wxID_ANY, 5, 1, 10, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	distanceToolbar->AddControl( stepSlider );
	invGeo = new wxCheckBox( distanceToolbar, wxID_ANY, wxT("Invert Geometry"), wxDefaultPosition, wxDefaultSize, 0 );
	distanceToolbar->AddControl( invGeo );
	distanceToolbar->Realize(); 
	
	bSizer5->Add( distanceToolbar, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	scanToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER|wxNO_BORDER ); 
	scanToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	scanToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	scanToolbar->Hide();
	
	scanVertEnable = new wxCheckBox( scanToolbar, wxID_ANY, wxT("Scanline vertical correction factor: "), wxDefaultPosition, wxDefaultSize, 0 );
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
	
	bSizer5->Add( scanToolbar, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	gainToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER|wxNO_BORDER ); 
	gainToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	gainToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	gainToolbar->Hide();
	
	useGain = new wxCheckBox( gainToolbar, wxID_ANY, wxT("Use Gain Correction"), wxDefaultPosition, wxDefaultSize, 0 );
	useGain->SetValue(true); 
	gainToolbar->AddControl( useGain );
	exposureValue = new wxStaticText( gainToolbar, wxID_ANY, wxT("50"), wxDefaultPosition, wxDefaultSize, 0 );
	exposureValue->Wrap( -1 );
	exposureValue->Hide();
	
	gainToolbar->AddControl( exposureValue );
	exposureLabel = new wxStaticText( gainToolbar, wxID_ANY, wxT(" ms"), wxDefaultPosition, wxDefaultSize, 0 );
	exposureLabel->Wrap( -1 );
	exposureLabel->Hide();
	
	gainToolbar->AddControl( exposureLabel );
	resetExposure = new wxButton( gainToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	gainToolbar->AddControl( resetExposure );
	exposureSlider = new wxSlider( gainToolbar, wxID_ANY, 10, 5, 25, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	exposureSlider->Hide();
	
	gainToolbar->AddControl( exposureSlider );
	voltageValue = new wxStaticText( gainToolbar, wxID_ANY, wxT("70"), wxDefaultPosition, wxDefaultSize, 0 );
	voltageValue->Wrap( -1 );
	voltageValue->Hide();
	
	gainToolbar->AddControl( voltageValue );
	voltageLabel = new wxStaticText( gainToolbar, wxID_ANY, wxT("kV"), wxDefaultPosition, wxDefaultSize, 0 );
	voltageLabel->Wrap( -1 );
	voltageLabel->Hide();
	
	gainToolbar->AddControl( voltageLabel );
	resetVoltage = new wxButton( gainToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	gainToolbar->AddControl( resetVoltage );
	voltageSlider = new wxSlider( gainToolbar, wxID_ANY, 14, 10, 14, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	voltageSlider->Hide();
	
	gainToolbar->AddControl( voltageSlider );
	gainToolbar->Realize(); 
	
	bSizer5->Add( gainToolbar, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	metalToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER|wxNO_BORDER ); 
	metalToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	metalToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	metalToolbar->Hide();
	
	useMetal = new wxCheckBox( metalToolbar, wxID_ANY, wxT("Metal Artifact Correction over value: "), wxDefaultPosition, wxDefaultSize, 0 );
	metalToolbar->AddControl( useMetal );
	metalValue = new wxStaticText( metalToolbar, wxID_ANY, wxT("8000"), wxDefaultPosition, wxDefaultSize, 0 );
	metalValue->Wrap( -1 );
	metalValue->Enable( false );
	metalValue->Hide();
	
	metalToolbar->AddControl( metalValue );
	resetMetal = new wxButton( metalToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	resetMetal->Enable( false );
	
	metalToolbar->AddControl( resetMetal );
	metalSlider = new wxSlider( metalToolbar, wxID_ANY, 16, 1, 50, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	metalSlider->Enable( false );
	metalSlider->Hide();
	
	metalToolbar->AddControl( metalSlider );
	metalToolbar->Realize(); 
	
	bSizer5->Add( metalToolbar, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	noiseToolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_NODIVIDER ); 
	noiseToolbar->SetForegroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_CAPTIONTEXT ) );
	noiseToolbar->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_MENU ) );
	noiseToolbar->Hide();
	
	outlierEnable = new wxCheckBox( noiseToolbar, wxID_ANY, wxT("Large noise hard removal over value:  "), wxDefaultPosition, wxDefaultSize, 0 );
	noiseToolbar->AddControl( outlierEnable );
	noiseMaxVal = new wxStaticText( noiseToolbar, wxID_ANY, wxT("700"), wxDefaultPosition, wxDefaultSize, 0 );
	noiseMaxVal->Wrap( -1 );
	noiseToolbar->AddControl( noiseMaxVal );
	resetNoiseMax = new wxButton( noiseToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	noiseToolbar->AddControl( resetNoiseMax );
	noiseMaxSlider = new wxSlider( noiseToolbar, wxID_ANY, 700, 300, 800, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	noiseToolbar->AddControl( noiseMaxSlider );
	TVEnable = new wxCheckBox( noiseToolbar, wxID_ANY, wxT("TV denoising with lambda: "), wxDefaultPosition, wxDefaultSize, 0 );
	noiseToolbar->AddControl( TVEnable );
	lambdaVal = new wxStaticText( noiseToolbar, wxID_ANY, wxT("2"), wxDefaultPosition, wxDefaultSize, 0 );
	lambdaVal->Wrap( -1 );
	noiseToolbar->AddControl( lambdaVal );
	resetLambda = new wxButton( noiseToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxDefaultSize, 0 );
	noiseToolbar->AddControl( resetLambda );
	lambdaSlider = new wxSlider( noiseToolbar, wxID_ANY, 2, 1, 20, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	noiseToolbar->AddControl( lambdaSlider );
	iterLabel = new wxStaticText( noiseToolbar, wxID_ANY, wxT("TV iterations: "), wxDefaultPosition, wxDefaultSize, 0 );
	iterLabel->Wrap( -1 );
	noiseToolbar->AddControl( iterLabel );
	iterVal = new wxStaticText( noiseToolbar, wxID_ANY, wxT("20"), wxDefaultPosition, wxDefaultSize, 0 );
	iterVal->Wrap( -1 );
	noiseToolbar->AddControl( iterVal );
	resetIter = new wxButton( noiseToolbar, wxID_ANY, wxT("Reset"), wxDefaultPosition, wxDefaultSize, 0 );
	noiseToolbar->AddControl( resetIter );
	iterSlider = new wxSlider( noiseToolbar, wxID_ANY, 20, 1, 50, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
	noiseToolbar->AddControl( iterSlider );
	noiseToolbar->Realize(); 
	
	bSizer5->Add( noiseToolbar, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer6->Add( bSizer5, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer7;
	bSizer7 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer7->Add( 0, 0, 1, 0, 5 );
	
	m_staticText35 = new wxStaticText( this, wxID_ANY, wxT("Recontruction preview (scroll to change distance):"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText35->Wrap( -1 );
	bSizer7->Add( m_staticText35, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	ok = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer7->Add( ok, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	cancel = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer7->Add( cancel, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer6->Add( bSizer7, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer6 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	this->Connect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( reconConfig::onClose ) );
	optionBox->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( reconConfig::onToolbarChoice ), NULL, this );
	distance->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( reconConfig::onDistance ), NULL, this );
	setStartDis->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onSetStartDis ), NULL, this );
	setEndDis->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onSetEndDis ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	invGeo->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onInvGeo ), NULL, this );
	scanVertEnable->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onScanVertEnable ), NULL, this );
	resetScanVert->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanHorEnable->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onScanHorEnable ), NULL, this );
	resetScanHor->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	useGain->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onEnableGain ), NULL, this );
	resetExposure->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	resetVoltage->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	useMetal->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onEnableMetal ), NULL, this );
	resetMetal->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	outlierEnable->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onNoiseMaxEnable ), NULL, this );
	resetNoiseMax->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	TVEnable->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onTVEnable ), NULL, this );
	resetLambda->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetLambda ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	resetIter->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetIter ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	ok->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onOk ), NULL, this );
	cancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onCancel ), NULL, this );
}

reconConfig::~reconConfig()
{
	// Disconnect Events
	this->Disconnect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( reconConfig::onClose ) );
	optionBox->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( reconConfig::onToolbarChoice ), NULL, this );
	distance->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( reconConfig::onDistance ), NULL, this );
	setStartDis->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onSetStartDis ), NULL, this );
	setEndDis->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onSetEndDis ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	stepSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onStepSlider ), NULL, this );
	invGeo->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onInvGeo ), NULL, this );
	scanVertEnable->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onScanVertEnable ), NULL, this );
	resetScanVert->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanVertSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onScanVert ), NULL, this );
	scanHorEnable->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onScanHorEnable ), NULL, this );
	resetScanHor->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	scanHorSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onScanHor ), NULL, this );
	useGain->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onEnableGain ), NULL, this );
	resetExposure->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	exposureSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onExposure ), NULL, this );
	resetVoltage->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	voltageSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onVoltage ), NULL, this );
	useMetal->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onEnableMetal ), NULL, this );
	resetMetal->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	metalSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onMetal ), NULL, this );
	outlierEnable->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onNoiseMaxEnable ), NULL, this );
	resetNoiseMax->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	noiseMaxSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onNoiseMax ), NULL, this );
	TVEnable->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( reconConfig::onTVEnable ), NULL, this );
	resetLambda->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetLambda ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	lambdaSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onLambdaSlider ), NULL, this );
	resetIter->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onResetIter ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	iterSlider->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( reconConfig::onIterSlider ), NULL, this );
	ok->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onOk ), NULL, this );
	cancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( reconConfig::onCancel ), NULL, this );
	
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

sliceDialog::sliceDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxGridBagSizer* gbSizer1;
	gbSizer1 = new wxGridBagSizer( 0, 0 );
	gbSizer1->SetFlexibleDirection( wxBOTH );
	gbSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	slicesLabel = new wxStaticText( this, wxID_ANY, wxT("Number of slices to save:"), wxDefaultPosition, wxDefaultSize, 0 );
	slicesLabel->Wrap( -1 );
	gbSizer1->Add( slicesLabel, wxGBPosition( 0, 0 ), wxGBSpan( 1, 1 ), wxALL, 5 );
	
	sliceValue = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	gbSizer1->Add( sliceValue, wxGBPosition( 0, 1 ), wxGBSpan( 1, 1 ), wxALL, 5 );
	
	
	this->SetSizer( gbSizer1 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	sliceValue->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( sliceDialog::onSliceValue ), NULL, this );
}

sliceDialog::~sliceDialog()
{
	// Disconnect Events
	sliceValue->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( sliceDialog::onSliceValue ), NULL, this );
	
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
	
	m_staticText35 = new wxStaticText( this, wxID_ANY, wxT("Reconstruction Iterations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText35->Wrap( -1 );
	fgSizer2->Add( m_staticText35, 0, wxALL, 5 );
	
	iterations = new wxTextCtrl( this, wxID_ANY, wxT("75"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( iterations, 0, wxALL, 5 );
	
	rawCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Raw images from detector"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( rawCheckBox, 0, wxALL, 5 );
	
	m_panel2 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	fgSizer2->Add( m_panel2, 1, wxEXPAND | wxALL, 5 );
	
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
