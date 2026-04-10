import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ChatService } from './chat.service';
import { BlogSection } from './chat.models';
import { ChatApiResponse, ChatMessage } from './chat.models';
import { HealthResponse } from './chat.models';
import { MetricsResponse } from './chat.models';
import { ReportDetail } from './chat.models';
import { ReportSummary } from './chat.models';

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  title = 'Editorial Sentinel';
  prompt = '';
  darkMode = false;
  referenceImageUrl = '';
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
  deletingReportId: string | null = null;
  activeMenu: 'new' | 'history' | 'saved' | 'settings' = 'new';
  activeTopTab: 'dashboard' | 'templates' | 'analytics' = 'templates';
  selectedReport: ReportDetail | null = null;
  healthData: HealthResponse | null = null;
  metricsData: MetricsResponse | null = null;
  metricsError = '';
  private pollTimerId: number | null = null;

  readonly heroImageUrl =
    'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=1600&q=80';
  readonly heroImageAlt = 'Team working in a modern digital workspace';

  readonly samplePrompts: string[] = [
    'Generate an in-depth editorial regarding police check procedures and background verification policies in the modern workplace.',
    'Create a concise version focused on executive readers.',
    'Rewrite in a confident, informative tone for HR leaders.'
  ];

  readonly messages: ChatMessage[] = [];
  readonly reports: ReportSummary[] = [];

  @ViewChild('promptInput')
  promptInput?: ElementRef<HTMLTextAreaElement>;

  constructor(private readonly chatService: ChatService) {
    this.refreshRuntime();
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
  }

  useSample(prompt: string): void {
    this.prompt = prompt;
  }

  setMenu(menu: 'new' | 'history' | 'saved' | 'settings'): void {
    this.activeMenu = menu;
    if (menu === 'history' || menu === 'saved') {
      this.loadReports();
    }
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

  sendPrompt(): void {
    const trimmed = this.prompt.trim();
    if (!trimmed || this.loading) {
      return;
    }

    const finalPrompt = this.buildPromptWithImage(this.buildConfiguredPrompt(trimmed));
    const imageUrl = this.referenceImageUrl.trim();
    const userText = imageUrl ? `${trimmed}\n\n[reference_image] ${imageUrl}` : trimmed;

    this.errorMessage = '';
    this.messages.push({ role: 'user', text: userText });
    this.loading = true;

    this.chatService.sendMessage(finalPrompt, this.sessionId).subscribe({
      next: (response: ChatApiResponse) => {
        this.sessionId = response.session_id;
        this.generatedAt = new Date();
        this.selectedReport = null;
        this.messages.push({
          role: 'assistant',
          text: response.generated.draft,
          payload: response,
        });
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.errorMessage = 'Cannot reach the API. Check that the backend server is running at http://localhost:8000.';
      }
    });
  }

  onPromptKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      this.sendPrompt();
    }
  }

  applyImagePromptTemplate(): void {
    const imageUrl = this.referenceImageUrl.trim() || 'https://images.example.com/reference.jpg';
    this.prompt = [
      'Write a blog section based on this visual direction.',
      'Focus on trust, compliance, and modern digital onboarding.',
      `Reference image: ${imageUrl}`,
      'Tone: professional, confident, clear.'
    ].join('\n');
    this.focusPrompt();
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

  copyDraft(response: ChatApiResponse): void {
    navigator.clipboard
      .writeText(response.generated.draft)
      .catch(() => {
        this.errorMessage = 'Unable to copy the draft. Please copy it manually.';
      });
  }

  usePromptTemplate(sample: string): void {
    this.prompt = sample;
    this.focusPrompt();
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

  private buildPromptWithImage(basePrompt: string): string {
    const imageUrl = this.referenceImageUrl.trim();
    if (!imageUrl) {
      return basePrompt;
    }

    return [
      basePrompt,
      '',
      'Use this image as visual reference for style and context:',
      `- image_url: ${imageUrl}`,
      '- keep visual language modern and trustworthy',
      '- maintain realistic details and sharp descriptive cues'
    ].join('\n');
  }

  exportDraft(response: ChatApiResponse): void {
    const content = [
      `Title: ${response.generated.title}`,
      '',
      'Outline:',
      ...response.generated.outline.map((item, index) => `${index + 1}. ${item}`),
      '',
      'Draft:',
      response.generated.draft,
      '',
      'Sources Used:',
      ...response.generated.sources_used,
    ].join('\n');

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `draft-${Date.now()}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }
}
