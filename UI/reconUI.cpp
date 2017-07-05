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
	m_mgr.SetManagedWindow(this);
	m_mgr.SetFlags(wxAUI_MGR_DEFAULT);
	
	m_menubar1 = new wxMenuBar( 0 );
	file = new wxMenu();
	wxMenuItem* newPage;
	newPage = new wxMenuItem( file, wxID_NEW, wxString( wxT("New\tCtrl+N") ) , wxEmptyString, wxITEM_NORMAL );
	file->Append( newPage );
	
	wxMenuItem* open;
	open = new wxMenuItem( file, wxID_OPEN, wxString( wxT("Open\tCtrl+O") ) , wxEmptyString, wxITEM_NORMAL );
	file->Append( open );
	
	wxMenuItem* quit;
	quit = new wxMenuItem( file, wxID_ANY, wxString( wxT("Exit\tAlt-X") ) , wxEmptyString, wxITEM_NORMAL );
	file->Append( quit );
	
	m_menubar1->Append( file, wxT("File") ); 
	
	config = new wxMenu();
	wxMenuItem* configDialog;
	configDialog = new wxMenuItem( config, wxID_PREFERENCES, wxString( wxT("Edit Config") ) , wxEmptyString, wxITEM_NORMAL );
	config->Append( configDialog );
	
	m_menubar1->Append( config, wxT("Config") ); 
	
	help = new wxMenu();
	wxMenuItem* about;
	about = new wxMenuItem( help, wxID_ABOUT, wxString( wxT("About\tF1") ) , wxEmptyString, wxITEM_NORMAL );
	help->Append( about );
	
	m_menubar1->Append( help, wxT("Help") ); 
	
	this->SetMenuBar( m_menubar1 );
	
	m_auinotebook6 = new wxAuiNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxAUI_NB_DEFAULT_STYLE );
	m_mgr.AddPane( m_auinotebook6, wxAuiPaneInfo() .Left() .CaptionVisible( false ).CloseButton( false ).PaneBorder( false ).Dock().Resizable().FloatingSize( wxDefaultSize ).CentrePane() );
	
	m_panel10 = new wxPanel( m_auinotebook6, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL );
	
	m_textCtrl8 = new wxTextCtrl( m_panel10, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer1->Add( m_textCtrl8, 1, wxALL|wxEXPAND, 5 );
	
	
	m_panel10->SetSizer( bSizer1 );
	m_panel10->Layout();
	bSizer1->Fit( m_panel10 );
	m_auinotebook6->AddPage( m_panel10, wxT("Start Here"), false, wxNullBitmap );
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	
	m_mgr.Update();
	this->Centre( wxBOTH );
	
	// Connect Events
	this->Connect( newPage->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onNew ) );
	this->Connect( open->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onOpen ) );
	this->Connect( quit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onQuit ) );
	this->Connect( configDialog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onConfig ) );
	this->Connect( about->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAbout ) );
}

mainWindow::~mainWindow()
{
	// Disconnect Events
	this->Disconnect( wxID_NEW, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onNew ) );
	this->Disconnect( wxID_OPEN, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onOpen ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onQuit ) );
	this->Disconnect( wxID_PREFERENCES, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onConfig ) );
	this->Disconnect( wxID_ABOUT, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( mainWindow::onAbout ) );
	
	m_mgr.UnInit();
	
}

configDialog::configDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	this->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_WINDOW ) );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 0, 4, 0, 0 );
	fgSizer2->SetFlexibleDirection( wxVERTICAL );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText1 = new wxStaticText( this, wxID_ANY, wxT("Automatically detected object distance"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1->Wrap( 200 );
	fgSizer2->Add( m_staticText1, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString generateDistanceChoices[] = { wxT("Yes"), wxT("No") };
	int generateDistanceNChoices = sizeof( generateDistanceChoices ) / sizeof( wxString );
	generateDistance = new wxRadioBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, generateDistanceNChoices, generateDistanceChoices, 1, wxRA_SPECIFY_COLS );
	generateDistance->SetSelection( 1 );
	fgSizer2->Add( generateDistance, 0, wxALL|wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticText2 = new wxStaticText( this, wxID_ANY, wxT("Estimated distance from detector to object"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText2->Wrap( 200 );
	fgSizer2->Add( m_staticText2, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	estimatedDistance = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( estimatedDistance, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticText4 = new wxStaticText( this, wxID_ANY, wxT("Number of slices to reconstruct"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4->Wrap( 200 );
	fgSizer2->Add( m_staticText4, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	reconstructionSlices = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( reconstructionSlices, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticText3 = new wxStaticText( this, wxID_ANY, wxT("Thickness of reconstruction slice"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText3->Wrap( 200 );
	fgSizer2->Add( m_staticText3, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	sliceThickness = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer2->Add( sliceThickness, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticText7 = new wxStaticText( this, wxID_ANY, wxT("Edge Blurring"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText7->Wrap( 200 );
	fgSizer2->Add( m_staticText7, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString edgeBlurEnabledChoices[] = { wxT("Enabled"), wxT("Disabled") };
	int edgeBlurEnabledNChoices = sizeof( edgeBlurEnabledChoices ) / sizeof( wxString );
	edgeBlurEnabled = new wxRadioBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, edgeBlurEnabledNChoices, edgeBlurEnabledChoices, 1, wxRA_SPECIFY_COLS );
	edgeBlurEnabled->SetSelection( 1 );
	fgSizer2->Add( edgeBlurEnabled, 1, wxALL|wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticText8 = new wxStaticText( this, wxID_ANY, wxT("TV denoising"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText8->Wrap( 200 );
	fgSizer2->Add( m_staticText8, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString denosingEnabledChoices[] = { wxT("Enabled"), wxT("Disabled") };
	int denosingEnabledNChoices = sizeof( denosingEnabledChoices ) / sizeof( wxString );
	denosingEnabled = new wxRadioBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, denosingEnabledNChoices, denosingEnabledChoices, 1, wxRA_SPECIFY_COLS );
	denosingEnabled->SetSelection( 1 );
	fgSizer2->Add( denosingEnabled, 1, wxALL|wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticText6 = new wxStaticText( this, wxID_ANY, wxT("Orientation"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6->Wrap( 200 );
	fgSizer2->Add( m_staticText6, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString orientationChoices[] = { wxT("Right"), wxT("Left") };
	int orientationNChoices = sizeof( orientationChoices ) / sizeof( wxString );
	orientation = new wxRadioBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, orientationNChoices, orientationChoices, 1, wxRA_SPECIFY_COLS );
	orientation->SetSelection( 1 );
	fgSizer2->Add( orientation, 1, wxALL|wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticText5 = new wxStaticText( this, wxID_ANY, wxT("Rotate Direction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText5->Wrap( 200 );
	fgSizer2->Add( m_staticText5, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString rotationEnabledChoices[] = { wxT("Yes"), wxT("No") };
	int rotationEnabledNChoices = sizeof( rotationEnabledChoices ) / sizeof( wxString );
	rotationEnabled = new wxRadioBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, rotationEnabledNChoices, rotationEnabledChoices, 1, wxRA_SPECIFY_COLS );
	rotationEnabled->SetSelection( 1 );
	fgSizer2->Add( rotationEnabled, 1, wxALL|wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticText13 = new wxStaticText( this, wxID_ANY, wxT("Detector information"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText13->Wrap( -1 );
	fgSizer2->Add( m_staticText13, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxGridSizer* gSizer2;
	gSizer2 = new wxGridSizer( 0, 2, 0, 0 );
	
	m_staticText9 = new wxStaticText( this, wxID_ANY, wxT("Height (pixels)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText9->Wrap( -1 );
	gSizer2->Add( m_staticText9, 0, wxALL, 5 );
	
	pixelWidth = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	gSizer2->Add( pixelWidth, 0, wxALL, 5 );
	
	m_staticText10 = new wxStaticText( this, wxID_ANY, wxT("Width (pixels)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText10->Wrap( -1 );
	gSizer2->Add( m_staticText10, 0, wxALL, 5 );
	
	pixelHeight = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	gSizer2->Add( pixelHeight, 0, wxALL, 5 );
	
	m_staticText11 = new wxStaticText( this, wxID_ANY, wxT("Pitch height"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText11->Wrap( -1 );
	gSizer2->Add( m_staticText11, 0, wxALL, 5 );
	
	pitchHeight = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	gSizer2->Add( pitchHeight, 0, wxALL, 5 );
	
	m_staticText12 = new wxStaticText( this, wxID_ANY, wxT("Pitch width"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText12->Wrap( -1 );
	gSizer2->Add( m_staticText12, 0, wxALL, 5 );
	
	pitchWidth = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
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
	loadConfig->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onLoad ), NULL, this );
	saveConfig->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onSave ), NULL, this );
	ok->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onOK ), NULL, this );
	cancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onCancel ), NULL, this );
}

configDialog::~configDialog()
{
	// Disconnect Events
	loadConfig->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onLoad ), NULL, this );
	saveConfig->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onSave ), NULL, this );
	ok->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onOK ), NULL, this );
	cancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( configDialog::onCancel ), NULL, this );
	
}
