import { Pipe, PipeTransform } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Pipe({ name: 'markdown', standalone: true })
export class MarkdownPipe implements PipeTransform {
  constructor(private readonly sanitizer: DomSanitizer) {}

  transform(value: string | null | undefined): SafeHtml {
    if (!value) {
      return '';
    }
    const html = this.markdownToHtml(value);
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  private markdownToHtml(md: string): string {
    let html = md;

    // ── Strip raw source citations completely ──
    // [Source: Title | URL: url] → remove entirely
    html = html.replace(
      /\s*\[Source:\s*[^\]]+?\]\s*/gi,
      ''
    );

    // ── Strip repetitive inline citation links ──
    // Remove patterns like [Apply for a National Police Check](url) that repeat
    // Keep the link text but strip the link if it appears more than twice
    const linkCounts: Record<string, number> = {};
    html = html.replace(
      /\[([^\]]+?)\]\((https?:\/\/[^)]+)\)/g,
      (match, text, _url) => {
        linkCounts[text] = (linkCounts[text] || 0) + 1;
        if (linkCounts[text] > 2) {
          return text; // Strip repeated links, keep text only
        }
        return match;
      }
    );

    // ── Pre-process: ensure ## headings have line breaks before them ──
    // Only target markdown headings (## or ###) that are glued to text without a newline
    html = html.replace(/([^\n])(#{2,4}\s+\S)/g, '$1\n$2');

    // ── Horizontal rules ──
    html = html.replace(/^---+$/gm, '<hr />');

    // ── Code blocks (fenced) ──
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, lang, code) => {
      const escaped = this.escapeHtml(code.trim());
      return `<pre><code class="lang-${lang || 'text'}">${escaped}</code></pre>`;
    });

    // ── Inline code ──
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // ── Headings ──
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // ── Clean up stray lone '#' markers left by LLM output ──
    html = html.replace(/^#{1,4}\s*$/gm, '');

    // ── Images ──
    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" loading="lazy" />');

    // ── Links ──
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // ── Bold and italic ──
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(?<!\w)\*(.+?)\*(?!\w)/g, '<em>$1</em>');

    // ── Blockquotes (multi-line support) ──
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
    // Merge adjacent blockquotes
    html = html.replace(/<\/blockquote>\n<blockquote>/g, '<br />');

    // ── Ordered lists ──
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<oli>$1</oli>');
    html = html.replace(/(<oli>.*<\/oli>\n?)+/g, (match) => {
      const items = match.replace(/<\/?oli>/g, (tag) =>
        tag === '<oli>' ? '<li>' : '</li>'
      );
      return `<ol>${items}</ol>`;
    });

    // ── Unordered lists ──
    html = html.replace(/^[-•] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
      // Don't double-wrap if already in <ol>
      if (match.includes('<ol>')) return match;
      return `<ul>${match}</ul>`;
    });

    // ── Paragraphs — wrap loose lines ──
    html = html.replace(/^(?!<[a-z])((?!^\s*$).+)$/gm, '<p>$1</p>');

    // ── Clean up double-wrapped paragraphs ──
    html = html.replace(/<p><(h[1-4]|ul|ol|li|pre|blockquote|img|hr)/g, '<$1');
    html = html.replace(/<\/(h[1-4]|ul|ol|li|pre|blockquote)><\/p>/g, '</$1>');

    return html;
  }

  private escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
}
