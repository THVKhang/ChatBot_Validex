import { AfterViewInit, Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ChatService } from './chat.service';
import { ChatApiResponse, ChatMessage } from './chat.models';

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

  readonly heroImageUrl =
    'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=1600&q=80';
  readonly heroImageAlt = 'Team working in a modern digital workspace';

  readonly samplePrompts: string[] = [
    'Generate an in-depth editorial regarding police check procedures and background verification policies in the modern workplace.',
    'Create a concise version focused on executive readers.',
    'Rewrite in a confident, informative tone for HR leaders.'
  ];

  readonly messages: ChatMessage[] = [];

  @ViewChild('promptInput')
  promptInput?: ElementRef<HTMLTextAreaElement>;

  constructor(private readonly chatService: ChatService) {
    this.chatService.healthCheck().subscribe({
      next: () => {
        this.apiReady = true;
      },
      error: () => {
        this.apiReady = false;
        this.errorMessage = 'Backend API is not running. Start the API server before testing the frontend.';
      }
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
  }

  useSample(prompt: string): void {
    this.prompt = prompt;
  }

  toggleDarkMode(): void {
    this.darkMode = !this.darkMode;
  }

  clearSession(): void {
    this.messages.splice(0, this.messages.length);
    this.sessionId = null;
    this.errorMessage = '';
    this.generatedAt = null;
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
