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
bool MyApp::OnInit()
{
	//set name for config files in registry
	SetVendorName(wxT("Xinvivo"));

	// call the base class initialization method, parses command line inputs
	if (!wxApp::OnInit())
		return false;

	// create the main application window
	DTRMainWindow *frame = new DTRMainWindow(NULL);
	frame->Show(true);

	wxStreamToTextRedirector redirect(frame->m_textCtrl8);
	//std::cout << "Test\n";
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

void DTRMainWindow::onQuit(wxCommandEvent& WXUNUSED(event)){
	// true is to force the frame to close
	Close(true);
}

void DTRMainWindow::onConfig(wxCommandEvent& WXUNUSED(event)) {
	DTRConfigDialog *frame2 = new DTRConfigDialog(NULL);
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

GLFrame::GLFrame(wxAuiNotebook *frame, const wxPoint& pos, const wxSize& size, long style)
	: wxPanel(frame, wxID_ANY, pos, size), m_canvas(NULL){
	//Set up sizer to make the canvas take up the entire panel (wxWidgets handles garbage collection)
	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer(wxVERTICAL);

	//initialize the canvas to this object
	m_canvas = new CudaGLCanvas(this, wxID_ANY, NULL, GetClientSize());

	bSizer2->Add(m_canvas, 1, wxEXPAND | wxALL, 5);
	this->SetSizer(bSizer2);
	this->Layout();
	bSizer2->Fit(this);

	// Show the frame
	Show(true);
	Raise();//grab attention when the frame has finished rendering
}

GLFrame::~GLFrame()
{
	delete m_canvas;
}

//---------------------------------------------------------------------------
// CudaGLCanvas
//---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(CudaGLCanvas, wxGLCanvas)
EVT_PAINT(CudaGLCanvas::OnPaint)
EVT_CHAR(CudaGLCanvas::OnChar)
EVT_MOUSE_EVENTS(CudaGLCanvas::OnMouseEvent)
wxEND_EVENT_TABLE()

CudaGLCanvas::CudaGLCanvas(wxWindow *parent, wxWindowID id, int* gl_attrib, wxSize size)
	: wxGLCanvas(parent, id, gl_attrib, wxDefaultPosition, size, wxFULL_REPAINT_ON_RESIZE){

	// Explicitly create a new rendering context instance for this canvas.
	m_glRC = new wxGLContext(this);
	//wxGLContextAttrs cxtArrs;
	//cxtArrs.CoreProfile().OGLVersion(4, 5).Robust().ResetIsolation().EndList();

	SetCurrent(*m_glRC);

	int argc = 1;
	char* argv[1] = { (char*)wxString((wxTheApp->argv)[0]).ToUTF8().data() };

	//wxStreamToTextRedirector redirect(((mainWindow*)this->GetParent()->GetParent())->m_textCtrl8);

	m_inter = new interop(&argc, argv, GetSize().x, GetSize().y, first);
	first = false;

	cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
	cudaEventCreateWithFlags(&event, cudaEventBlockingSync);

	/*TomoError retErr = TomoRecon();
	if(retErr != Tomo_OK)
		wxMessageBox(wxT("Reconstruction error!"),
			wxString::Format(wxT("Reconstruction failed with error code : %d"),(int)retErr),
			wxICON_ERROR | wxOK);*/
}

CudaGLCanvas::~CudaGLCanvas(){
	delete m_inter;
	delete m_glRC;
}

void CudaGLCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)){
	// This is a dummy, to avoid an endless succession of paint messages.
	// OnPaint handlers must always create a wxPaintDC.
	wxPaintDC(this);

	SetCurrent(*m_glRC);//tells opengl which buffers to use, mutliple windows fail without this
	m_inter->display(GetSize().x, GetSize().y);
	m_inter->map(stream);

	pxl_kernel_launcher(*(m_inter->ca),
		GetSize().x,
		GetSize().y,
		event,
		stream);

	m_inter->unmap(stream);

	m_inter->blit();
	m_inter->swap();

	SwapBuffers();
}

void CudaGLCanvas::OnChar(wxKeyEvent& event){
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