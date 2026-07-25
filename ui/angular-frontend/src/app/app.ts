import { Component, ElementRef, ViewChild } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
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
  title = 'Validex AI';
  prompt = '';
  darkMode = false;
  selectedTone = 'Professional';
  selectedWordCount = '500 Words';
  selectedSectionCount = '3';
  selectedImageCount = '1';
  selectedAudience = 'General Audience';
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
  tokenUsageData: any = null;
  isDiscovering = false;
  discoveryResult = '';
  showAuthModal = false;
  authMode: 'login' | 'register' = 'login';
  authUsername = '';
  authPassword = '';
  authError = '';
  authSuccess = '';
  authLoading = false;

  // HITL Dashboard State
  pendingReviews: import('./chat.models').PendingReview[] = [];
  pendingReviewsLoading = false;
  selectedReview: import('./chat.models').PendingReview | null = null;

  // OS Dock State
  dockPopup: string | null = null;
  showTokenPopup = false;
  showSettings = false;
  // Module 1: Feedback State
  feedbackMap: Record<number, 1 | -1> = {};

  // Module 2: Analytics State
  analyticsTokens: any = null;
  analyticsQuality: any = null;
  analyticsCache: any = null;
  analyticsFeedback: any = null;

  // Module 3: Language State
  selectedLanguage = 'Auto';
  detectedLanguage = '';
  readonly languageOptions = [
    { code: 'Auto', label: '🌐 Auto-detect' },
    { code: 'en', label: '🇬🇧 English' },
    { code: 'vi', label: '🇻🇳 Tiếng Việt' },
    { code: 'zh', label: '🇨🇳 中文' },
    { code: 'ko', label: '🇰🇷 한국어' },
    { code: 'ja', label: '🇯🇵 日本語' },
  ];

  selectedValidexVersion = '1.3';
  readonly validexVersions = ['1.1', '1.2', '1.3'];

  // Custom Blog Settings (free input)
  customWordCount: number = 500;
  customSectionCount: number = 3;
  customImageCount: number = 1;

  // Module 4: Pipeline Visualizer State
  pipelineSteps: { name: string; icon: string; status: 'done' | 'active' | 'pending'; detail: string }[] = [];

  // Module 5: Scheduling State
  schedules: any[] = [];
  newScheduleTopic = '';
  newScheduleLang = 'en';
  newScheduleCron = '0 9 * * MON';

  // Module 6: Supervisor State
  lastComplexityLevel: 'simple' | 'complex' | '' = '';
  supervisorNotes = '';

  // User Token Budget State
  userBudget: { remaining: number; total: number; used: number; percent: number; tier: string; requests: number } | null = null;
  quotaExceeded = false;

  toggleDockPopup(name: string): void {
    this.dockPopup = this.dockPopup === name ? null : name;
  }

  closeDockPopups(): void {
    this.dockPopup = null;
  }

  thinkingStatus = '';
  thinkingDetail = '';
  private thinkingInterval: any;
  
  isTyping = false;
  private typingInterval: any;

  private startThinkingCycle(): void {
    this.thinkingStatus = 'Initializing Multi-Agent system...';
    // Initialize pipeline steps for visualizer
    this.pipelineSteps = [
      { name: 'Parser', icon: 'psychology', status: 'pending', detail: '' },
      { name: 'Researcher', icon: 'search', status: 'pending', detail: '' },
      { name: 'RAG Eval', icon: 'fact_check', status: 'pending', detail: '' },
      { name: 'Supervisor', icon: 'hub', status: 'pending', detail: '' },
      { name: 'Writer', icon: 'edit_note', status: 'pending', detail: '' },
      { name: 'Editor', icon: 'rate_review', status: 'pending', detail: '' },
    ];
    this.lastComplexityLevel = '';
    this.supervisorNotes = '';
    this.detectedLanguage = '';
  }

  private stopThinkingCycle(): void {
    this.thinkingStatus = '';
    this.thinkingDetail = '';
  }

  private updatePipelineStep(nodeName: string, detail: string): void {
    const nodeMap: Record<string, string> = {
      'Parser': 'Parser', 'Researcher': 'Researcher',
      'RAG_Evaluator': 'RAG Eval', 'Supervisor': 'Supervisor',
      'Deep_Researcher': 'Researcher', 'Writer': 'Writer', 'Editor': 'Editor',
    };
    const stepName = nodeMap[nodeName] || nodeName;
    let found = false;
    for (const step of this.pipelineSteps) {
      if (step.name === stepName) {
        step.status = 'active';
        step.detail = detail;
        found = true;
      } else if (found) {
        step.status = 'pending';
      } else {
        step.status = 'done';
      }
    }
    // Special: insert Deep_Researcher after Supervisor when complex
    if (nodeName === 'Deep_Researcher') {
      const hasDR = this.pipelineSteps.some(s => s.name === 'Deep Research');
      if (!hasDR) {
        const supIdx = this.pipelineSteps.findIndex(s => s.name === 'Supervisor');
        if (supIdx >= 0) {
          this.pipelineSteps.splice(supIdx + 1, 0, {
            name: 'Deep Research', icon: 'library_books', status: 'active', detail: detail,
          });
        }
      }
    }
  }

  private typewriterEffect(targetMsg: ChatMessage, fullText: string): void {
    targetMsg.text = fullText;
    this.scrollChatToBottom();
  }

  private readonly allPrompts: string[] = [
    'How to apply for a police check in Australia — step by step guide',
    'Police check requirements for employers hiring in aged care and childcare',
    'What volunteers need to know about police checks in Australia',
    'National Police Check vs. Working With Children Check — key differences explained',
    'How long does a police check take in Australia? Processing times by state',
    'Police check requirements for visa applications and immigration to Australia',
    'Do police checks expire? Understanding validity periods for employers',
    'How to get a police check in NSW, VIC, and QLD — state-by-state breakdown',
    'Criminal history disclosure: What shows up on an Australian police check?',
    'Why employers in healthcare and education must conduct police checks',
    'Online vs. in-person police checks: Which option is right for you?',
    'ACIC accreditation explained: How Validex delivers trusted police checks',
  ];

  readonly samplePrompts: string[] = this.getRandomPrompts(3);

  private getRandomPrompts(count: number): string[] {
    const shuffled = [...this.allPrompts].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count);
  }

  readonly messages: ChatMessage[] = [];
  readonly reports: ReportSummary[] = [];

  @ViewChild('promptInput')
  promptInput?: ElementRef<HTMLTextAreaElement>;

  @ViewChild('chatHistory')
  chatHistory?: ElementRef<HTMLDivElement>;

  private _sanitizer!: DomSanitizer;
  private _markdownPipeInstance!: MarkdownPipe;

  constructor(private readonly chatService: ChatService, private authService: AuthService, sanitizer: DomSanitizer) {
    this._sanitizer = sanitizer;
    this._markdownPipeInstance = new MarkdownPipe(sanitizer);
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
    this.loadTokenUsage();
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
      this.loadTokenUsage();
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
        console.error('Backend API is not running. Start the API server before testing the frontend.');
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

  loadTokenUsage(): void {
    this.chatService.getTokenUsage().subscribe({
      next: (res) => {
        this.tokenUsageData = res;
      },
      error: () => {
        this.tokenUsageData = null;
      }
    });
    this.loadUserBudget();
  }

  loadUserBudget(): void {
    this.chatService.getUserBudget().subscribe({
      next: (res) => {
        this.userBudget = res;
        this.quotaExceeded = res.remaining <= 0;
      },
      error: () => {
        this.userBudget = null;
        this.quotaExceeded = false;
      }
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
    this.chatService.getTokenUsage().subscribe({
      next: (res) => {
        this.tokenUsageData = res;
      },
      error: () => {
        this.tokenUsageData = null;
      }
    });
    this.chatService.getCrawlHistory().subscribe({
      next: (res) => {
        this.crawlHistory = res.history || res.logs || [];
      },
      error: () => {
        this.crawlHistory = [];
      }
    });
    this.loadPendingReviews();
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
            const status = event.data?.status || '';
            const detail = event.data?.detail || '';
            const stepName = event.data?.step || 'Agent working';
            
            // Update Pipeline Visualizer (Module 4)
            this.updatePipelineStep(stepName, detail || status);

            // Track Supervisor routing (Module 6)
            if (stepName === 'Supervisor') {
              this.lastComplexityLevel = status.toLowerCase().includes('complex') ? 'complex' : 'simple';
              this.supervisorNotes = detail || status;
            }

            if (status) {
              this.thinkingStatus = status;
              this.thinkingDetail = detail;
            } else {
              if (stepName === 'Parser') this.thinkingStatus = '🎯 Analyzing your request...';
              else if (stepName === 'Researcher') this.thinkingStatus = '🔍 Searching knowledge...';
              else if (stepName === 'Supervisor') this.thinkingStatus = '🧠 Routing pipeline...';
              else if (stepName === 'Deep_Researcher') this.thinkingStatus = '📚 Deep-diving legal sources...';
              else if (stepName === 'Writer') this.thinkingStatus = '✍️ Generating content...';
              else if (stepName === 'Editor') this.thinkingStatus = '🔬 Reviewing quality...';
              else this.thinkingStatus = `${stepName} is working...`;
              this.thinkingDetail = '';
            }
          } else if (event.type === 'chunk') {
            this.stopThinkingCycle();
            this.isTyping = true;
            assistantMsg.text += (event.data?.chunk || '');
            this.scrollChatToBottom();
          } else if (event.type === 'done') {
            this.stopThinkingCycle();
            const response = event.data as ChatApiResponse;
            this.sessionId = response.session_id;
            this.generatedAt = new Date();
            this.selectedReport = null;
            assistantMsg.payload = response;
            assistantMsg.latencyMs = Date.now() - startTime;
            this.loading = false;
            // Mark all pipeline steps as done
            for (const s of this.pipelineSteps) s.status = 'done';
            // Track detected language (Module 3)
            this.detectedLanguage = response.parsed?.language || 'en';
            // Trigger smooth typewriter UI
            this.isTyping = false;
            if (!assistantMsg.text) {
              assistantMsg.text = response.generated.draft;
            }
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
    if (event.key === 'Enter' && !event.shiftKey) {
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

  loadPendingReviews(): void {
    this.pendingReviewsLoading = true;
    this.chatService.getPendingReviews().subscribe({
      next: (reviews) => {
        this.pendingReviewsLoading = false;
        this.pendingReviews = reviews;
      },
      error: () => {
        this.pendingReviewsLoading = false;
        // Optionally show error
      }
    });
  }

  approveReview(runId: string): void {
    this.chatService.updateReview(runId, { action: 'Approve' }).subscribe(() => {
      this.loadPendingReviews();
    });
  }

  rejectReview(runId: string): void {
    this.chatService.updateReview(runId, { action: 'Reject', feedback: 'Rejected by Editor' }).subscribe(() => {
      this.loadPendingReviews();
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

  exportDraft(markdown: string, format: 'docx' | 'pdf' | 'html'): void {
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

  // ── Rich Text Editor State ──
  editorToolbarVisible: Record<number, boolean> = {};
  editableMode: Record<number, boolean> = {};
  currentTextColor = '#000000';
  currentHighlightColor = '#ffff00';
  currentFontSizeIndex = 3; // maps to fontSize "3" = 12pt
  pageOrientation: 'portrait' | 'landscape' = 'portrait';
  pageSize: 'a4' | 'letter' | 'legal' = 'a4';
  private _selectedImage: HTMLImageElement | null = null;

  setPageOrientation(orientation: 'portrait' | 'landscape'): void {
    this.pageOrientation = orientation;
  }

  setPageSize(size: string): void {
    this.pageSize = size as 'a4' | 'letter' | 'legal';
  }

  onEditableClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;

    // ── Image click: select and show resize handles ──
    if (target.tagName === 'IMG') {
      event.preventDefault();
      this._selectImage(target as HTMLImageElement);
    } else {
      this._deselectImage();
    }
  }

  private _selectImage(img: HTMLImageElement): void {
    this._deselectImage(); // clear previous
    this._selectedImage = img;
    img.classList.add('img-selected');

    // Create resize handle
    const handle = document.createElement('div');
    handle.className = 'img-resize-handle';
    handle.contentEditable = 'false';

    // Position handle relative to image
    const wrapper = document.createElement('span');
    wrapper.className = 'img-resize-wrapper';
    wrapper.contentEditable = 'false';
    img.parentNode?.insertBefore(wrapper, img);
    wrapper.appendChild(img);
    wrapper.appendChild(handle);

    // Drag to resize
    let startX = 0;
    let startWidth = 0;

    const onMouseDown = (e: MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      startX = e.clientX;
      startWidth = img.offsetWidth;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };

    const onMouseMove = (e: MouseEvent) => {
      const dx = e.clientX - startX;
      const newWidth = Math.max(50, startWidth + dx);
      img.style.width = newWidth + 'px';
      img.style.height = 'auto';
    };

    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    handle.addEventListener('mousedown', onMouseDown);
  }

  private _deselectImage(): void {
    if (this._selectedImage) {
      this._selectedImage.classList.remove('img-selected');
      // Unwrap from resize wrapper
      const wrapper = this._selectedImage.closest('.img-resize-wrapper');
      if (wrapper && wrapper.parentNode) {
        wrapper.parentNode.insertBefore(this._selectedImage, wrapper);
        wrapper.remove();
      }
      this._selectedImage = null;
    }
  }

  toggleEditorToolbar(msgIndex: number): void {
    const wasVisible = this.editorToolbarVisible[msgIndex];
    this.editorToolbarVisible[msgIndex] = !wasVisible;

    if (!wasVisible) {
      // ENTERING edit mode: capture rendered HTML from display div BEFORE Angular hides it
      const displayEl = document.getElementById(`msg-display-${msgIndex}`);
      const capturedHtml = displayEl ? displayEl.innerHTML : '';

      this.editableMode[msgIndex] = true;

      // Wait for Angular to render the edit div, then populate and focus
      setTimeout(() => {
        const editEl = document.getElementById(`msg-content-${msgIndex}`);
        if (editEl) {
          editEl.innerHTML = capturedHtml;
          editEl.focus();
        }
      }, 0);
    } else {
      // EXITING edit mode: save edits back to message text (optional)
      const editEl = document.getElementById(`msg-content-${msgIndex}`);
      if (editEl) {
        // Store edited HTML so it renders correctly in display mode
        const msg = this.messages[msgIndex];
        if (msg) {
          // Save the raw edited HTML back — the markdown pipe won't re-process it,
          // but the display div will show the updated content via innerHTML
          (msg as any)._editedHtml = editEl.innerHTML;
        }
      }
      this.editableMode[msgIndex] = false;
    }
  }

  isEditorToolbarVisible(msgIndex: number): boolean {
    return !!this.editorToolbarVisible[msgIndex];
  }

  isContentEditable(msgIndex: number): boolean {
    return !!this.editableMode[msgIndex];
  }

  execFormatCommand(command: string, value: string = ''): void {
    document.execCommand(command, false, value || undefined);
  }

  formatHeading(level: string): void {
    if (level) {
      document.execCommand('formatBlock', false, level);
    }
  }

  changeFontSize(delta: number): void {
    this.currentFontSizeIndex = Math.max(1, Math.min(7, this.currentFontSizeIndex + delta));
    document.execCommand('fontSize', false, String(this.currentFontSizeIndex));
  }

  applyTextColor(color: string): void {
    this.currentTextColor = color;
    document.execCommand('foreColor', false, color);
  }

  applyHighlightColor(color: string): void {
    this.currentHighlightColor = color;
    document.execCommand('hiliteColor', false, color);
  }

  insertLink(): void {
    const url = prompt('Enter URL:', 'https://');
    if (url) {
      document.execCommand('createLink', false, url);
    }
  }

  insertImage(): void {
    const url = prompt('Enter image URL:', 'https://');
    if (url) {
      document.execCommand('insertImage', false, url);
    }
  }

  getEditedContent(msgIndex: number): string {
    const el = document.querySelector(`#msg-content-${msgIndex}`) as HTMLElement;
    return el ? el.innerHTML : '';
  }

  // Used by display div to show edited HTML if user has edited, otherwise original markdown
  getDisplayHtml(msg: any): any {
    if (msg._editedHtml) {
      return this._sanitizer.bypassSecurityTrustHtml(msg._editedHtml);
    }
    return this._markdownPipeInstance.transform(msg.text || '');
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
    const lines = [
      basePrompt,
      '',
      'Editorial settings:',
      `- tone: ${this.selectedTone}`,
      `- target_word_count: ${this.selectedWordCount}`,
      `- target_audience: ${this.selectedAudience}`,
    ];

    // Language injection (Module 3)
    if (this.selectedLanguage !== 'Auto') {
      lines.push(`- language: ${this.selectedLanguage}`);
    }

    if (this.selectedSectionCount !== 'Auto') {
      const secCount = parseInt(this.selectedSectionCount);
      if (!isNaN(secCount)) {
        lines.push(`- target_sections: ${secCount}`);
      }
    }

    if (this.selectedImageCount !== 'Auto') {
      const imgCount = parseInt(this.selectedImageCount);
      if (!isNaN(imgCount)) {
        lines.push(`- target_images: ${imgCount}`);
      } else if (this.selectedImageCount === 'No images') {
        lines.push(`- target_images: 0`);
      }
    }

    return lines.join('\n');
  }

  // --- Auth Methods ---
  openAuthModal(mode: 'login' | 'register') {
    this.authMode = mode;
    this.showAuthModal = true;
    this.authError = '';
    this.authSuccess = '';
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
    this.authSuccess = '';

    if (this.authMode === 'login') {
      this.authService.login(this.authUsername, this.authPassword).subscribe({
        next: () => {
          this.authLoading = false;
          this.closeAuthModal();
          this.loadSessions();
        },
        error: (err) => {
          this.authLoading = false;
          this.authError = err?.error?.detail || 'Login failed. Check your credentials.';
        }
      });
    } else {
      this.authService.register(this.authUsername, this.authPassword).subscribe({
        next: () => {
          this.authLoading = false;
          this.authSuccess = '✅ Registration successful! Switching to login...';
          this.authError = '';
          // Auto-switch to login after 1.5 seconds
          setTimeout(() => {
            this.authMode = 'login';
            this.authSuccess = 'Account created! Please sign in.';
            this.authPassword = '';
          }, 1500);
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

  // ── Module 1: Feedback Methods ──
  submitFeedback(msgIndex: number, rating: 1 | -1): void {
    if (this.feedbackMap[msgIndex] !== undefined) return;
    const msg = this.messages[msgIndex];
    const reportId = msg?.payload?.session_id || this.sessionId || 'unknown';
    this.feedbackMap[msgIndex] = rating;
    this.chatService.submitFeedback(reportId, rating).subscribe({
      error: () => { delete this.feedbackMap[msgIndex]; }
    });
  }

  // ── Module 2: Analytics Methods ──
  loadAnalytics(): void {
    this.chatService.getAnalyticsTokens().subscribe({ next: (d) => this.analyticsTokens = d, error: () => {} });
    this.chatService.getAnalyticsQuality().subscribe({ next: (d) => this.analyticsQuality = d, error: () => {} });
    this.chatService.getAnalyticsCache().subscribe({ next: (d) => this.analyticsCache = d, error: () => {} });
    this.chatService.getAnalyticsFeedback().subscribe({ next: (d) => this.analyticsFeedback = d, error: () => {} });
  }

  get qualityEntries(): { key: string; value: number }[] {
    const data = this.analyticsQuality?.data;
    if (!data) return [];
    return Object.entries(data).map(([k, v]) => ({ key: k, value: v as number })).sort((a, b) => b.value - a.value);
  }

  get qualityTotal(): number {
    return this.qualityEntries.reduce((sum, e) => sum + e.value, 0);
  }

  // ── Module 3: Language Helper ──
  getLanguageLabel(code: string): string {
    return this.languageOptions.find(l => l.code === code)?.label || code;
  }

  // ── Module 5: Scheduling Methods ──
  loadSchedules(): void {
    this.chatService.getSchedules().subscribe({
      next: (s) => this.schedules = s || [],
      error: () => this.schedules = [],
    });
  }

  createSchedule(): void {
    if (!this.newScheduleTopic.trim()) return;
    this.chatService.createSchedule(this.newScheduleTopic, this.newScheduleLang, this.newScheduleCron).subscribe({
      next: () => {
        this.newScheduleTopic = '';
        this.loadSchedules();
      },
      error: () => {}
    });
  }

  deleteSchedule(id: number): void {
    this.chatService.deleteSchedule(id).subscribe({
      next: () => this.loadSchedules(),
      error: () => {}
    });
  }
}
