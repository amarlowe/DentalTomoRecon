#include "UIMain.h"

// Define a new application type, each program should derive a class from wxApp
class MyApp : public wxApp
{
public:
	virtual bool OnInit() wxOVERRIDE;
};

//Thread entry point, causes onInit to run like a main function would.
wxIMPLEMENT_APP(MyApp);

//Main equivalent: the program execution starts here
bool MyApp::OnInit(){
	//set name for config files in registry
	SetVendorName(wxT("Xinvivo"));

	// call the base class initialization method, parses command line inputs
	if (!wxApp::OnInit())
		return false;

	int argc = 1;
	char* argv[1] = { (char*)wxString((wxTheApp->argv)[0]).ToUTF8().data() };
	reconGlutInit(&argc, argv);

	// create the main application window
	DTRMainWindow *frame = new DTRMainWindow(NULL);

#ifdef PROFILER
	static unsigned s_pageAdded = 0;
	frame->m_auinotebook6->AddPage(frame->CreateNewPage(),
		wxString::Format
		(
			wxT("%u"),
			++s_pageAdded
		),
		true);
#else
	frame->Show(true);
#endif

	wxStreamToTextRedirector redirect(frame->m_textCtrl8);
	
	/*TomoError retErr = TomoRecon();
	if (retErr != Tomo_OK)
		wxMessageBox(wxT("Reconstruction error!"),
			wxString::Format(wxT("Reconstruction failed with error code : %d"), (int)retErr),
			wxICON_ERROR | wxOK);*/

	return true;
}

// ----------------------------------------------------------------------------
// main frame
// ----------------------------------------------------------------------------

DTRMainWindow::DTRMainWindow(wxWindow* parent) : mainWindow(parent){
	wxConfigBase *pConfig = wxConfigBase::Get();
	if (pConfig == NULL)
		return;

	//size edits
	if (pConfig->Read(wxT("/dialog/max"), 0l) == 1)
		wxTopLevelWindow::Maximize(true);
	else {
		int x = pConfig->Read(wxT("/dialog/x"), 50),
			y = pConfig->Read(wxT("/dialog/y"), 50),
			w = pConfig->Read(wxT("/dialog/w"), 350),
			h = pConfig->Read(wxT("/dialog/h"), 200);
		Move(x, y);
		SetClientSize(w, h);
	}
}

// event handlers
void DTRMainWindow::onNew(wxCommandEvent& WXUNUSED(event)) {
	static unsigned s_pageAdded = 0;
	m_auinotebook6->AddPage(CreateNewPage(),
		wxString::Format
		(
			wxT("%u"),
			++s_pageAdded
		),
		true);
}

wxPanel *DTRMainWindow::CreateNewPage() const{
	return new GLFrame(m_auinotebook6);
}

void DTRMainWindow::onOpen(wxCommandEvent& WXUNUSED(event)) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRMainWindow::onSave(wxCommandEvent& WXUNUSED(event)) {
	//TODO: add error checking
	saveThread* thd = new saveThread(((GLFrame*)(m_auinotebook6->GetCurrentPage()))->m_canvas->recon, m_statusBar1);
	thd->Create();
	thd->Run();
}

void DTRMainWindow::onQuit(wxCommandEvent& WXUNUSED(event)){
	// true is to force the frame to close
	Close(true);
}

void DTRMainWindow::onStep(wxCommandEvent& WXUNUSED(event)) {
	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	switch (recon->currentDisplay) {
	case raw_images:
		recon->correctProjections();
		recon->currentDisplay = sino_images;
		break;
	case sino_images:
		recon->currentDisplay = raw_images2;
		break;
	case raw_images2:
		recon->reconInit();
		currentFrame->m_scrollBar->SetScrollbar(0, 1, recon->Sys->Recon->Nz, 1);
		recon->currentDisplay = norm_images;
		break;
	case norm_images:
		recon->currentDisplay = recon_images;//intentionally skipped break
	case recon_images:
		recon->reconStep();
		break;
	}

	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onContinue(wxCommandEvent& WXUNUSED(event)) {
	//Run the entire reconstruction
	//Swtich statement is to make it state aware, but otherwise finishes out whatever is left
	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	ReconThread* thd = new ReconThread(currentFrame->m_canvas, recon, currentFrame, m_statusBar1);
	thd->Create();
	thd->Run();
}

void DTRMainWindow::onContRun(wxCommandEvent& WXUNUSED(event)) {
	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;
	RunBox* progress = new RunBox(this);
	progress->m_gauge2->SetRange(30);
	progress->m_gauge2->SetValue(10);
	progress->Show(true);

	switch (recon->currentDisplay) {
	case raw_images:
		recon->correctProjections();
	case sino_images:
	case raw_images2:
		recon->reconInit();
		currentFrame->m_scrollBar->SetScrollbar(0, 1, recon->Sys->Recon->Nz, 1);
	case norm_images:
		recon->currentDisplay = recon_images;
	case recon_images:
		while (recon->iteration < 30) {
			recon->reconStep();
			progress->m_gauge2->SetValue(recon->iteration + 1);
			currentFrame->m_canvas->paint();
		}
	}
	delete progress;

	saveThread* thd = new saveThread(recon, m_statusBar1);
	thd->Create();
	thd->Run();
}

void DTRMainWindow::onConfig(wxCommandEvent& WXUNUSED(event)) {
	DTRConfigDialog *frame2 = new DTRConfigDialog(this);
	frame2->Show(true);
}

void DTRMainWindow::onAbout(wxCommandEvent& WXUNUSED(event)){
	wxMessageBox(wxString::Format
	(
		"Welcome to Xinvivo's reconstruction app!\n"
		"\n"
		"This app was built for %s.",
		wxGetOsDescription()
	),
		"About Tomography Reconstruction",
		wxOK | wxICON_INFORMATION,
		this);
}

DTRMainWindow::~DTRMainWindow() {
	wxConfigBase *pConfig = wxConfigBase::Get();
	if (pConfig == NULL)
		return;

	// save the frame position
	int x, y, w, h;
	GetClientSize(&w, &h);
	GetPosition(&x, &y);
	pConfig->Write(wxT("/dialog/x"), (long)x);
	pConfig->Write(wxT("/dialog/y"), (long)y);
	pConfig->Write(wxT("/dialog/w"), (long)w);
	pConfig->Write(wxT("/dialog/h"), (long)h);
	if (wxTopLevelWindow::IsMaximized())
		pConfig->Write(wxT("/dialog/max"), 1);
	else
		pConfig->Write(wxT("/dialog/max"), 0);

	cuda(DeviceReset());//only reset here where we know all windows are finished
}

// ----------------------------------------------------------------------------
// Config frame handling
// ----------------------------------------------------------------------------

DTRConfigDialog::DTRConfigDialog(wxWindow* parent) : configDialog(parent){
	//load all values from previously saved settings
	wxConfigBase *pConfig = wxConfigBase::Get();

	estimatedDistance->SetValue(wxString::Format(wxT("%.1f"), pConfig->ReadDouble(wxT("/estimatedDistance"), 5.0f)));
	reconstructionSlices->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/reconstructionSlices"), 45l)));
	sliceThickness->SetValue(wxString::Format(wxT("%.1f"), pConfig->ReadDouble(wxT("/sliceThickness"), 0.5f)));
	pixelWidth->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/pixelWidth"), 1915l)));
	pixelHeight->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/pixelHeight"), 1440l)));
	pitchHeight->SetValue(wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxT("/pitchHeight"), 0.0185f)));
	pitchWidth->SetValue(wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxT("/pitchWidth"), 0.0185f)));
	for (int i = 0; i < m_grid1->GetNumberCols(); i++)
		for (int j = 0; j < m_grid1->GetNumberRows(); j++) 
			m_grid1->SetCellValue(j, i, wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxString::Format(wxT("/beamLoc%d-%d"),j,i), 0.0f)));
}

void DTRConfigDialog::onLoad(wxCommandEvent& event) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRConfigDialog::onSave(wxCommandEvent& event) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRConfigDialog::onOK(wxCommandEvent& event) {
	//check each value for invalid arguments
	//save each valid value to GUI config storage
	//TODO: also save each valid value to internal program storage

	//will not save all valid values, only until a bad value is hit
	double parsedDouble;
	long parsedInt = 0;
	wxConfigBase *pConfig = wxConfigBase::Get();
	if (!estimatedDistance->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Estimated distance from detector to object\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/estimatedDistance"), parsedDouble);

	if (!reconstructionSlices->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Number of slices to reconstruct\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/reconstructionSlices"), parsedInt);

	if (!sliceThickness->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Thickness of reconstruciton slice\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/sliceThickness"), parsedDouble);

	if (!pixelWidth->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Height (pixels)\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pixelWidth"), parsedInt);

	if (!pixelHeight->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Width (pixels)\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pixelHeight"), parsedInt);

	if (!pitchHeight->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Pitch height\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pitchHeight"), parsedDouble);

	if (!pitchWidth->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Pitch width\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pitchWidth"), parsedDouble);

	for (int i = 0; i < m_grid1->GetNumberCols(); i++) {
		for (int j = 0; j < m_grid1->GetNumberRows(); j++) {
			if (!m_grid1->GetCellValue(j, i).ToDouble(&parsedDouble)) {
				wxMessageBox(wxString::Format(wxT("Invalid input in text box: \"Beam emitter locations (%d,%d)\"."),j+1,i+1),//add one for average user readability
					wxT("Invlaid input"),
					wxICON_STOP | wxOK);
				return;
			}
			else pConfig->Write(wxString::Format(wxT("/beamLoc%d-%d"), j, i), parsedDouble);
		}
	}

	//All values are valid, set done flag and return
	//TODO: set done flag
	Close(true);
}


void DTRConfigDialog::onCancel(wxCommandEvent& WXUNUSED(event)) {
	Close(true);
}

DTRConfigDialog::~DTRConfigDialog() {
	
}

//---------------------------------------------------------------------------
// GLFrame
//---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(GLFrame, wxPanel)
EVT_SCROLL(GLFrame::OnScroll)
EVT_MOUSEWHEEL(GLFrame::OnMousewheel)
wxEND_EVENT_TABLE()

GLFrame::GLFrame(wxAuiNotebook *frame, const wxPoint& pos, const wxSize& size, long style)
	: wxPanel(frame, wxID_ANY, pos, size), m_canvas(NULL){
	//Set up sizer to make the canvas take up the entire panel (wxWidgets handles garbage collection)
	wxBoxSizer* bSizer;
	bSizer = new wxBoxSizer(wxVERTICAL);

	//initialize the canvas to this object
	m_canvas = new CudaGLCanvas(this, wxID_ANY, NULL, GetClientSize());
	bSizer->Add(m_canvas, 1, wxEXPAND | wxALL, 5);

	m_scrollBar = new wxScrollBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL);
	m_scrollBar->SetScrollbar(0, 1, 7, 1);
	bSizer->Add(m_scrollBar, 0, wxALL | wxEXPAND, 5);

	this->SetSizer(bSizer);
	this->Layout();
	bSizer->Fit(this);

	// Show the frame
	Show(true);
	Raise();//grab attention when the frame has finished rendering
}

GLFrame::~GLFrame(){
	delete m_canvas;
}

void GLFrame::OnScroll(wxScrollEvent& event) {
	m_canvas->OnScroll(m_scrollBar->GetThumbPosition());
}

void GLFrame::OnMousewheel(wxMouseEvent& event) {
	int newScrollPos =event.GetWheelRotation() / 120;
	newScrollPos += m_scrollBar->GetThumbPosition();
	if (newScrollPos < 0) newScrollPos = 0;
	if (newScrollPos > m_scrollBar->GetRange() - 1) newScrollPos = m_scrollBar->GetRange() - 1;
	m_scrollBar->SetThumbPosition(newScrollPos);
	m_canvas->OnScroll(newScrollPos);
}

//---------------------------------------------------------------------------
// CudaGLCanvas
//---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(CudaGLCanvas, wxGLCanvas)
EVT_PAINT(CudaGLCanvas::OnPaint)
EVT_CHAR(CudaGLCanvas::OnChar)
EVT_MOUSE_EVENTS(CudaGLCanvas::OnMouseEvent)
EVT_COMMAND(wxID_ANY, PAINT_IT, CudaGLCanvas::OnEvent)
wxEND_EVENT_TABLE()

CudaGLCanvas::CudaGLCanvas(wxWindow *parent, wxWindowID id, int* gl_attrib, wxSize size)
	: wxGLCanvas(parent, id, gl_attrib, wxDefaultPosition, size, wxFULL_REPAINT_ON_RESIZE){
	// Explicitly create a new rendering context instance for this canvas.
	m_glRC = new wxGLContext(this);

	SetCurrent(*m_glRC);

	recon = new TomoRecon(GetSize().x, GetSize().y);
	recon->init();

#ifdef PROFILER
	recon->correctProjections();
	recon->reconInit();
	recon->reconStep();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	exit(0);
	//paint();
#endif
}

CudaGLCanvas::~CudaGLCanvas(){
	delete recon;
	delete m_glRC;
}

void CudaGLCanvas::OnScroll(int index) {
	imageIndex = index;
	paint();
}

void CudaGLCanvas::OnEvent(wxCommandEvent& WXUNUSED(event)) {
	// This is a dummy, to avoid an endless succession of paint messages.
	// OnPaint handlers must always create a wxPaintDC.
	//wxPaintDC(this);

	if (recon->initialized) {
		paint();
	}
}

void CudaGLCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)){
	// This is a dummy, to avoid an endless succession of paint messages.
	// OnPaint handlers must always create a wxPaintDC.
	wxPaintDC(this);

	if (recon->initialized) {
		paint();
	}
}

void CudaGLCanvas::paint() {
	SetCurrent(*m_glRC);//tells opengl which buffers to use, mutliple windows fail without this
	int width = GetSize().x;
	int height = GetSize().y;
	recon->display(width, height);
	recon->map();

	recon->test(imageIndex);

	recon->unmap();

	recon->blit();
	recon->swap();

	SwapBuffers();
}

void CudaGLCanvas::OnChar(wxKeyEvent& event){
	//Moved to main window, can delete
}

void CudaGLCanvas::OnMouseEvent(wxMouseEvent& event){
	static int dragging = 0;
	static float last_x, last_y;

	// Allow default processing to happen, or else the canvas cannot gain focus
	// (for key events).
	event.Skip();

	if (event.LeftIsDown())
	{
		if (!dragging)
		{
			dragging = 1;
		}
		else
		{
			//m_yrot += (event.GetX() - last_x)*1.0;
			//m_xrot += (event.GetY() - last_y)*1.0;
			Refresh(false);
		}
		last_x = event.GetX();
		last_y = event.GetY();
	}
	else
	{
		dragging = 0;
	}
}

//---------------------------------------------------------------------------
// ReconThread
//---------------------------------------------------------------------------

DEFINE_EVENT_TYPE(PAINT_IT);
ReconThread::ReconThread(wxEvtHandler* pParent, TomoRecon* recon, GLFrame* Frame, wxStatusBar* status)
	: wxThread(wxTHREAD_DETACHED), m_pParent(pParent), status(status), currentFrame(Frame), m_recon(recon) {
}

wxThread::ExitCode ReconThread::Entry(){
	wxCommandEvent needsPaint(PAINT_IT, GetId());
	//Run the entire reconstruction
	//Swtich statement is to make it state aware, but otherwise finishes out whatever is left
	switch (m_recon->currentDisplay) {
	case raw_images:
		m_recon->correctProjections();
		//currentFrame->m_canvas->paint();
		wxPostEvent(m_pParent, needsPaint);
	case sino_images:
	case raw_images2:
		m_recon->reconInit();
	case norm_images:
		m_recon->currentDisplay = recon_images;
		wxMutexGuiEnter();
		currentFrame->m_scrollBar->SetScrollbar(0, 1, m_recon->Sys->Recon->Nz, 1);
		wxMutexGuiLeave();
	case recon_images:
		wxGauge* progress = new wxGauge(status, wxID_ANY, 30, wxPoint(100, 3));
		progress->SetValue(0);
		while (!TestDestroy() && m_recon->iteration < 30) {
			m_recon->reconStep();
			wxPostEvent(m_pParent, needsPaint);
			status->SetStatusText(wxT("Reconstructing:"));
			progress->SetValue(m_recon->iteration+1);
			this->Sleep(500);
			//this->Yield();
		}
		delete progress;
	}
	status->SetStatusText(wxT("Saving image..."));
	m_recon->TomoSave();
	status->SetStatusText(wxT("Image saved!"));
	
	return static_cast<ExitCode>(NULL);
}

//---------------------------------------------------------------------------
// saveThread
//---------------------------------------------------------------------------

saveThread::saveThread(TomoRecon* recon, wxStatusBar* status) : wxThread(wxTHREAD_DETACHED), m_recon(recon), status(status){
}

wxThread::ExitCode saveThread::Entry() {
	status->SetStatusText(wxT("Saving image..."));
	m_recon->TomoSave();
	status->SetStatusText(wxT("Image saved!"));

	return static_cast<ExitCode>(NULL);
}