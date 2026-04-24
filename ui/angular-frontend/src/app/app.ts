import { Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ChatService, StreamEvent } from './chat.service';
import { MarkdownPipe } from './markdown.pipe';
import { BlogSection } from './chat.models';
import { ChatApiResponse, ChatMessage } from './chat.models';
import { HealthResponse } from './chat.models';
import { KnowledgeHealthResponse } from './chat.models';
import { MetricsResponse } from './chat.models';
import { AuthorityBandStat } from './chat.models';
import { ReportDetail } from './chat.models';
import { ReportStatus } from './chat.models';
import { ReportSummary } from './chat.models';
import { SourceAnalyticsResponse } from './chat.models';
import { TopicChunkStat } from './chat.models';
import { ChatSessionSummary, ChatSessionHistory } from './chat.models';
import { AuthService } from './auth.service';

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule, MarkdownPipe],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  title = 'Editorial Sentinel';
  prompt = '';
  darkMode = false;
  selectedTone = 'Professional';
  selectedWordCount = '800 Words';
  selectedAudience = 'HR Professionals';
  sessionId: string | null = null;
  loading = false;
  apiReady = false;
  errorMessage = '';
  generatedAt: Date | null = null;
  reportsLoading = false;
  reportsError = '';
  savingReport = false;
  updatingReportStatus = false;
  deletingReportId: string | null = null;
  activeMenu: 'new' | 'history' | 'saved' | 'settings' | 'admin' = 'new';
  activeTopTab: 'dashboard' | 'templates' | 'analytics' = 'templates';

  sessionsLoading = false;
  sessionsError = '';
  sessions: ChatSessionSummary[] = [];
  selectedReport: ReportDetail | null = null;
  healthData: HealthResponse | null = null;
  metricsData: MetricsResponse | null = null;
  sourceAnalyticsData: SourceAnalyticsResponse | null = null;
  knowledgeHealthData: KnowledgeHealthResponse | null = null;
  metricsError = '';
  sourceAnalyticsError = '';
  knowledgeHealthError = '';
  useStreaming = true;
  private pollTimerId: number | null = null;
  attachedFile: string | null = null;
  attachedFileContent: string | null = null;
  isUploading = false;

  // Authentication State
  currentUser: string | null = null;
  isAdmin: boolean = false;
  adminUsers: any[] = [];
  crawlHistory: any[] = [];
  isDiscovering = false;
  discoveryResult = '';
  showAuthModal = false;
  authMode: 'login' | 'register' = 'login';
  authUsername = '';
  authPassword = '';
  authError = '';
  authLoading = false;

  thinkingStatus = '';
  private thinkingInterval: any;

  private startThinkingCycle(): void {
    // We now rely on real-time SSE "thinking" events from LangGraph multi-agent architecture
    // This is just the initial state
    this.thinkingStatus = 'Initializing Multi-Agent system...';
  }

  private stopThinkingCycle(): void {
    this.thinkingStatus = '';
  }

  readonly samplePrompts: string[] = [
    'Generate an in-depth editorial regarding police check procedures and background verification policies in the modern workplace.',
    'Create a concise version focused on executive readers.',
    'Rewrite in a confident, informative tone for HR leaders.'
  ];

  readonly messages: ChatMessage[] = [];
  readonly reports: ReportSummary[] = [];

  @ViewChild('promptInput')
  promptInput?: ElementRef<HTMLTextAreaElement>;

  @ViewChild('chatHistory')
  chatHistory?: ElementRef<HTMLDivElement>;

  constructor(private readonly chatService: ChatService, private authService: AuthService) {
    this.refreshRuntime();
    this.authService.currentUser$.subscribe(user => {
      this.currentUser = user;
    });
    this.authService.isAdmin$.subscribe(isAdmin => {
      this.isAdmin = isAdmin;
    });
  }

  ngAfterViewInit(): void {
    const params = new URLSearchParams(window.location.search);
    const promptFromUrl = params.get('prompt');
    const autoSend = params.get('autosend') === '1';

    if (promptFromUrl) {
      this.prompt = promptFromUrl;
    }

    this.focusPrompt();

    if (promptFromUrl && autoSend) {
      setTimeout(() => this.sendPrompt(), 150);
    }

    this.loadReports();
    this.startRuntimePolling();
  }

  ngOnDestroy(): void {
    if (this.pollTimerId !== null) {
      window.clearInterval(this.pollTimerId);
      this.pollTimerId = null;
    }
  }

  startRuntimePolling(): void {
    if (this.pollTimerId !== null) {
      window.clearInterval(this.pollTimerId);
    }

    this.pollTimerId = window.setInterval(() => {
      this.refreshRuntime();
    }, 10000);
  }

  refreshRuntime(): void {
    this.chatService.healthCheck().subscribe({
      next: (health) => {
        this.apiReady = true;
        this.healthData = health;
      },
      error: () => {
        this.apiReady = false;
        this.healthData = null;
        this.errorMessage = 'Backend API is not running. Start the API server before testing the frontend.';
      }
    });

    this.chatService.getMetrics().subscribe({
      next: (metrics) => {
        this.metricsData = metrics;
        this.metricsError = '';
      },
      error: () => {
        this.metricsData = null;
        this.metricsError = 'Unable to fetch runtime metrics.';
      },
    });

    this.chatService.getSourceAnalytics().subscribe({
      next: (analytics) => {
        this.sourceAnalyticsData = analytics;
        this.sourceAnalyticsError = '';
      },
      error: () => {
        this.sourceAnalyticsData = null;
        this.sourceAnalyticsError = 'Unable to fetch source analytics.';
      },
    });

    this.chatService.getKnowledgeHealth().subscribe({
      next: (health) => {
        this.knowledgeHealthData = health;
        this.knowledgeHealthError = '';
      },
      error: () => {
        this.knowledgeHealthData = null;
        this.knowledgeHealthError = 'Unable to fetch knowledge health.';
      },
    });
  }

  useSample(prompt: string): void {
    this.prompt = prompt;
  }

  setMenu(menu: 'new' | 'history' | 'saved' | 'settings' | 'admin'): void {
    this.activeMenu = menu;
    if (menu === 'history') {
      this.loadSessions();
    } else if (menu === 'saved') {
      this.loadReports();
    } else if (menu === 'admin') {
      this.loadAdminData();
    }
  }

  loadAdminData(): void {
    this.chatService.getAdminUsers().subscribe({
      next: (res) => {
        this.adminUsers = res.users || [];
      },
      error: (err) => {
        console.error('Failed to load admin users', err);
      }
    });
  }

  setTopTab(tab: 'dashboard' | 'templates' | 'analytics'): void {
    this.activeTopTab = tab;
  }

  toggleDarkMode(): void {
    this.darkMode = !this.darkMode;
  }

  clearSession(): void {
    this.messages.splice(0, this.messages.length);
    this.sessionId = null;
    this.errorMessage = '';
    this.generatedAt = null;
    this.selectedReport = null;
    this.activeMenu = 'new';
    this.focusPrompt();
  }

  focusPrompt(): void {
    setTimeout(() => this.promptInput?.nativeElement.focus(), 30);
  }

  scrollChatToBottom(): void {
    setTimeout(() => {
      const el = this.chatHistory?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    }, 50);
  }

  sendPrompt(): void {
    const trimmed = this.prompt.trim();
    if (!trimmed && !this.attachedFileContent || this.loading) {
      return;
    }

    let finalPrompt = this.buildConfiguredPrompt(trimmed);
    let userText = trimmed;

    if (this.attachedFileContent) {
      const fileContext = `\n\n[CONTEXT FROM ATTACHED FILE: ${this.attachedFile}]\n${this.attachedFileContent}\n\n`;
      finalPrompt += fileContext;
      userText = `[File Attached: ${this.attachedFile}]\n` + userText;
      
      this.attachedFile = null;
      this.attachedFileContent = null;
    }

    this.errorMessage = '';
    this.messages.push({ role: 'user', text: userText });
    this.prompt = '';
    this.loading = true;
    this.scrollChatToBottom();

    if (this.useStreaming) {
      this.sendViaStream(finalPrompt);
    } else {
      this.sendViaHttp(finalPrompt);
    }
  }

  private sendViaHttp(finalPrompt: string): void {
    this.chatService.sendMessage(finalPrompt, this.sessionId).subscribe({
      next: (response: ChatApiResponse) => {
        this.handleResponse(response);
      },
      error: () => {
        this.loading = false;
        this.errorMessage = 'Cannot reach the API. Check that the backend server is running at http://localhost:8000.';
      }
    });
  }

  private async sendViaStream(finalPrompt: string): Promise<void> {
    const startTime = Date.now();

    // Add a placeholder assistant message
    const assistantMsg: ChatMessage = {
      role: 'assistant',
      text: '',
    };
    this.messages.push(assistantMsg);
    this.scrollChatToBottom();
    this.startThinkingCycle();

    try {
      await this.chatService.sendMessageStream(
        finalPrompt,
        this.sessionId,
        (event: StreamEvent) => {
          if (event.type === 'meta') {
            // Update latency display
            const latency = event.data.latency_ms;
            if (latency) {
              assistantMsg.latencyMs = Math.round(latency);
            }
          } else if (event.type === 'thinking') {
            const stepName = event.data?.step || 'Agent working';
            if (stepName === 'Parser') this.thinkingStatus = 'Parsing prompt semantics...';
            else if (stepName === 'Researcher') this.thinkingStatus = 'Data Collector AI retrieving knowledge...';
            else if (stepName === 'Writer') this.thinkingStatus = 'Blog Writer AI generating draft...';
            else if (stepName === 'Editor') this.thinkingStatus = 'Editor AI reviewing compliance...';
            else this.thinkingStatus = `${stepName} is working...`;
          } else if (event.type === 'done') {
            this.stopThinkingCycle();
            const response = event.data as ChatApiResponse;
            this.sessionId = response.session_id;
            this.generatedAt = new Date();
            this.selectedReport = null;
            assistantMsg.text = response.generated.draft;
            assistantMsg.payload = response;
            assistantMsg.latencyMs = Date.now() - startTime;
            this.loading = false;
            this.scrollChatToBottom();
          } else if (event.type === 'error') {
            this.stopThinkingCycle();
            this.loading = false;
            this.errorMessage = event.data.error || 'Stream error';
          }
        }
      );
    } catch {
      this.stopThinkingCycle();
      this.loading = false;
      this.errorMessage = 'Cannot reach the API. Check that the backend server is running.';
    }

    if (this.loading) {
      this.stopThinkingCycle();
      this.loading = false;
    }
  }

  private handleResponse(response: ChatApiResponse): void {
    this.sessionId = response.session_id;
    this.generatedAt = new Date();
    this.selectedReport = null;
    this.messages.push({
      role: 'assistant',
      text: response.generated.draft,
      payload: response,
    });
    this.loading = false;
    this.scrollChatToBottom();
  }

  regenerateMessage(messageIndex: number): void {
    // Find the user message that prompted this assistant response
    let userPrompt = '';
    for (let i = messageIndex - 1; i >= 0; i--) {
      if (this.messages[i].role === 'user') {
        userPrompt = this.messages[i].text;
        break;
      }
    }
    if (!userPrompt || this.loading) return;

    // Remove the old assistant message
    this.messages.splice(messageIndex, 1);

    // Re-send
    this.loading = true;
    this.scrollChatToBottom();
    const finalPrompt = this.buildConfiguredPrompt(userPrompt);

    if (this.useStreaming) {
      this.sendViaStream(finalPrompt);
    } else {
      this.sendViaHttp(finalPrompt);
    }
  }

  onPromptKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      this.sendPrompt();
    }
  }

  saveLatestReport(): void {
    const payload = this.latestResponse;
    if (!payload || this.savingReport) {
      return;
    }

    const latestUserPrompt = this.latestUserPrompt;
    if (!latestUserPrompt) {
      this.errorMessage = 'No prompt found for the current draft.';
      return;
    }

    this.savingReport = true;
    this.reportsError = '';

    this.chatService
      .saveReport({
        session_id: this.sessionId,
        prompt: latestUserPrompt,
        generated: payload.generated,
      })
      .subscribe({
        next: ({ report }) => {
          this.savingReport = false;
          this.selectedReport = report;
          this.loadReports();
          this.activeMenu = 'saved';
        },
        error: () => {
          this.savingReport = false;
          this.reportsError = 'Unable to save the draft right now.';
        },
      });
  }

  loadReports(): void {
    this.reportsLoading = true;
    this.reportsError = '';
    this.chatService.listReports(30).subscribe({
      next: ({ reports }) => {
        this.reportsLoading = false;
        this.reports.splice(0, this.reports.length, ...reports);
      },
      error: () => {
        this.reportsLoading = false;
        this.reportsError = 'Unable to load saved reports.';
      },
    });
  }

  loadSessions(): void {
    this.sessionsLoading = true;
    this.sessionsError = '';
    this.chatService.getChatSessions(30).subscribe({
      next: (sessions) => {
        this.sessionsLoading = false;
        this.sessions.splice(0, this.sessions.length, ...sessions);
      },
      error: () => {
        this.sessionsLoading = false;
        this.sessionsError = 'Unable to load chat sessions.';
      },
    });
  }

  openSession(sessionId: string): void {
    this.sessionsError = '';
    this.chatService.getChatSession(sessionId).subscribe({
      next: (history: ChatSessionHistory) => {
        this.clearSession();
        this.sessionId = history.session_id;
        
        // Reconstruct messages from turns
        for (const turn of history.turns) {
          if (turn.user_prompt) {
            this.messages.push({ role: 'user', text: turn.user_prompt });
          }
          if (turn.assistant_output) {
            this.messages.push({ role: 'assistant', text: turn.assistant_output });
          }
        }
        
        this.activeMenu = 'new';
        this.scrollChatToBottom();
      },
      error: () => {
        this.sessionsError = 'Unable to load this session.';
      }
    });
  }

  openReport(reportId: string): void {
    this.reportsError = '';
    this.chatService.getReport(reportId).subscribe({
      next: ({ report }) => {
        this.selectedReport = report;
        this.generatedAt = report.created_at ? new Date(report.created_at) : null;
        this.activeMenu = 'saved';
      },
      error: () => {
        this.reportsError = 'Unable to open this report.';
      },
    });
  }

  deleteReport(reportId: string): void {
    if (this.deletingReportId) {
      return;
    }

    this.deletingReportId = reportId;
    this.reportsError = '';

    this.chatService.deleteReport(reportId).subscribe({
      next: () => {
        this.deletingReportId = null;
        const nextReports = this.reports.filter((item) => item.id !== reportId);
        this.reports.splice(0, this.reports.length, ...nextReports);

        if (this.selectedReport?.id === reportId) {
          this.selectedReport = null;
        }
      },
      error: () => {
        this.deletingReportId = null;
        this.reportsError = 'Unable to delete this report.';
      },
    });
  }

  updateSelectedReportStatus(status: ReportStatus): void {
    if (!this.selectedReport || this.updatingReportStatus) {
      return;
    }
    if (this.selectedReport.status === status) {
      return;
    }

    this.updatingReportStatus = true;
    this.reportsError = '';

    this.chatService.updateReportStatus(this.selectedReport.id, status).subscribe({
      next: ({ report }) => {
        this.updatingReportStatus = false;
        this.selectedReport = report;
        this.loadReports();
      },
      error: (err) => {
        this.updatingReportStatus = false;
        const detail = err?.error?.detail;
        this.reportsError = typeof detail === 'string' ? detail : 'Unable to update report status.';
      },
    });
  }

  copyDraft(response: ChatApiResponse): void {
    navigator.clipboard
      .writeText(response.generated.draft)
      .catch(() => {
        this.errorMessage = 'Unable to copy the draft. Please copy it manually.';
      });
  }

  exportDraft(markdown: string, format: 'docx' | 'html'): void {
    this.chatService.exportChat(markdown, format).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `validex_report.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
      },
      error: () => {
        this.errorMessage = `Unable to export to ${format.toUpperCase()}.`;
      }
    });
  }

  usePromptTemplate(sample: string): void {
    this.prompt = sample;
    this.focusPrompt();
  }

  isIngesting = false;
  ingestResult = '';

  checkIngestStatus(): void {
    this.chatService.getIngestStatus().subscribe({
      next: (res) => {
        if (res.running) {
          this.isIngesting = true;
          this.ingestResult = 'Ingestion is running...';
        } else {
          this.isIngesting = false;
        }
      },
      error: () => {}
    });
  }

  triggerIngest(): void {
    this.isIngesting = true;
    this.ingestResult = '';
    this.chatService.triggerIngest().subscribe({
      next: (res) => {
        this.isIngesting = false;
        this.ingestResult = res.message || 'Ingestion started/completed.';
      },
      error: (err) => {
        this.isIngesting = false;
        this.ingestResult = 'Error: ' + (err?.error?.detail || 'Unauthorized or failed.');
      }
    });
  }

  loadCrawlHistory(): void {
    this.chatService.getCrawlHistory().subscribe({
      next: (res) => { this.crawlHistory = res.logs || []; },
      error: () => { this.crawlHistory = []; }
    });
  }

  triggerDiscovery(): void {
    this.isDiscovering = true;
    this.discoveryResult = '';
    this.chatService.triggerDiscovery().subscribe({
      next: (res) => {
        this.isDiscovering = false;
        this.discoveryResult = res.message || 'Discovery completed.';
        this.loadCrawlHistory();
      },
      error: (err) => {
        this.isDiscovering = false;
        this.discoveryResult = 'Error: ' + (err?.error?.detail || 'Discovery failed.');
      }
    });
  }

  onFileSelected(event: any): void {
    const file = event.target.files?.[0];
    if (!file) return;

    this.isUploading = true;
    this.errorMessage = '';
    
    this.chatService.uploadFile(file).subscribe({
      next: (res) => {
        this.isUploading = false;
        this.attachedFile = res.filename;
        this.attachedFileContent = res.extracted_text;
      },
      error: (err) => {
        this.isUploading = false;
        this.errorMessage = 'Failed to extract text from file: ' + (err?.error?.detail || err.message);
      }
    });
  }

  get latestResponse(): ChatApiResponse | null {
    for (let index = this.messages.length - 1; index >= 0; index -= 1) {
      const payload = this.messages[index].payload;
      if (payload) {
        return payload;
      }
    }
    return null;
  }

  get draftedTimeLabel(): string {
    if (!this.generatedAt) {
      return 'No draft generated yet';
    }

    const minutes = Math.max(1, Math.floor((Date.now() - this.generatedAt.getTime()) / 60000));
    return `Drafted ${minutes} minute${minutes === 1 ? '' : 's'} ago`;
  }

  get latestUserPrompt(): string {
    for (let index = this.messages.length - 1; index >= 0; index -= 1) {
      const message = this.messages[index];
      if (message.role === 'user') {
        return message.text;
      }
    }
    return '';
  }

  get activeResultTitle(): string {
    if (this.selectedReport) {
      return this.selectedReport.title;
    }
    return this.latestResponse?.generated.title || '';
  }

  get activeResultOutline(): string[] {
    if (this.selectedReport) {
      return this.selectedReport.outline;
    }
    return this.latestResponse?.generated.outline || [];
  }

  get activeResultDraft(): string {
    if (this.selectedReport) {
      return this.selectedReport.draft;
    }
    return this.latestResponse?.generated.draft || '';
  }

  get activeResultSections(): BlogSection[] {
    if (this.selectedReport?.sections?.length) {
      return this.selectedReport.sections;
    }
    return this.latestResponse?.generated.sections || [];
  }

  sectionParagraphs(body: string): string[] {
    return body
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
  }

  get hasResult(): boolean {
    return Boolean(this.selectedReport || this.latestResponse);
  }

  get activeReportStatus(): ReportStatus | '' {
    return this.selectedReport?.status || '';
  }

  get canMarkReviewed(): boolean {
    return this.selectedReport?.status === 'Draft';
  }

  get canMarkApproved(): boolean {
    return this.selectedReport?.status === 'Reviewed';
  }

  get runtimeRetrievalMode(): string {
    return this.healthData?.runtime?.retrieval_mode || 'unknown';
  }

  get runtimeGenerationMode(): string {
    return this.healthData?.runtime?.generation_mode || 'unknown';
  }

  get runtimeQualityGateEnabled(): boolean {
    return Boolean(this.healthData?.runtime?.quality_gate_enabled);
  }

  get metricsAvgLatencyMs(): number {
    return this.metricsData?.latency?.avg_ms || 0;
  }

  get metricsP95LatencyMs(): number {
    return this.metricsData?.latency?.p95_ms || 0;
  }

  get qualityGateBlockedRate(): string {
    const total = this.metricsData?.chat_requests_total || 0;
    const blocked = this.metricsData?.quality_gate_blocked_total || 0;
    if (!total) {
      return '0%';
    }
    return `${Math.round((blocked / total) * 100)}%`;
  }

  modeEntries(map: Record<string, number> | undefined): Array<{ key: string; value: number }> {
    if (!map) {
      return [];
    }

    return Object.entries(map)
      .map(([key, value]) => ({ key, value }))
      .sort((a, b) => b.value - a.value);
  }

  get sourceTopicStats(): TopicChunkStat[] {
    return this.sourceAnalyticsData?.topics || [];
  }

  get sourceAuthorityStats(): AuthorityBandStat[] {
    return this.sourceAnalyticsData?.authority_bands || [];
  }

  get maxTopicChunks(): number {
    return Math.max(0, ...this.sourceTopicStats.map((item) => item.chunks));
  }

  get maxAuthoritySources(): number {
    return Math.max(0, ...this.sourceAuthorityStats.map((item) => item.sources));
  }

  get knowledgeGenuinePercent(): number {
    return this.knowledgeHealthData?.genuine_percent || 0;
  }

  get knowledgeFakePercent(): number {
    return this.knowledgeHealthData?.fake_percent || 0;
  }

  get knowledgeOtherPercent(): number {
    return this.knowledgeHealthData?.other_percent || 0;
  }

  get knowledgeReadyForRetrieval(): boolean {
    return Boolean(this.knowledgeHealthData?.ready_for_retrieval);
  }

  percentWidth(value: number): string {
    const bounded = Math.max(0, Math.min(100, Math.round(value * 100) / 100));
    return `${bounded}%`;
  }

  barWidth(value: number, maxValue: number): string {
    if (maxValue <= 0) {
      return '0%';
    }
    const percent = Math.round((value / maxValue) * 100);
    return `${Math.max(8, percent)}%`;
  }

  private buildConfiguredPrompt(basePrompt: string): string {
    return [
      basePrompt,
      '',
      'Editorial settings:',
      `- tone: ${this.selectedTone}`,
      `- target_word_count: ${this.selectedWordCount}`,
      `- target_audience: ${this.selectedAudience}`,
    ].join('\n');
  }

  // --- Auth Methods ---
  openAuthModal(mode: 'login' | 'register') {
    this.authMode = mode;
    this.showAuthModal = true;
    this.authError = '';
    this.authUsername = '';
    this.authPassword = '';
  }

  closeAuthModal() {
    this.showAuthModal = false;
  }

  submitAuth() {
    if (!this.authUsername || !this.authPassword) {
      this.authError = 'Please enter username and password.';
      return;
    }

    this.authLoading = true;
    this.authError = '';

    if (this.authMode === 'login') {
      this.authService.login(this.authUsername, this.authPassword).subscribe({
        next: () => {
          this.authLoading = false;
          this.closeAuthModal();
          this.loadSessions(); // Reload sessions for the logged-in user
        },
        error: (err) => {
          this.authLoading = false;
          this.authError = err?.error?.detail || 'Login failed. Check your credentials.';
        }
      });
    } else {
      this.authService.register(this.authUsername, this.authPassword).subscribe({
        next: () => {
          // Auto login after register
          this.authService.login(this.authUsername, this.authPassword).subscribe({
            next: () => {
              this.authLoading = false;
              this.closeAuthModal();
              this.loadSessions();
            },
            error: () => {
              this.authLoading = false;
              this.authMode = 'login';
              this.authError = 'Registration successful. Please log in.';
            }
          });
        },
        error: (err) => {
          this.authLoading = false;
          this.authError = err?.error?.detail || 'Registration failed. Username might be taken.';
        }
      });
    }
  }

  logout() {
    this.authService.logout();
    this.sessionId = null;
    this.messages.length = 0;
    this.sessions = [];
    this.activeMenu = 'new';
  }
}
