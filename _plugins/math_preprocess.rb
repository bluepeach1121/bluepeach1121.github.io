# frozen_string_literal: true

require "jekyll"
require "cgi"

module Jekyll
  class MathPreprocess < Generator
    safe true
    priority :high

    # Protect things in which dollar signs should NOT be interpreted as math:
    # - fenced code blocks
    # - inline code
    # - existing $$ ... $$ math
    # - raw <pre> / <code> blocks
    PROTECT_RE = %r{
      (?:^\s*```[^\n]*\n.*?^\s*```\s*$)
      |
      (?:^\s*~~~[^\n]*\n.*?^\s*~~~\s*$)
      |
      (?:\$\$[\s\S]*?\$\$)
      |
      (?:`[^`\n]*`)
      |
      (?i:<pre[^>]*>.*?</pre>)
      |
      (?i:<code[^>]*>.*?</code>)
    }mx

    # Obsidian-style inline math: $ ... $
    #
    # Opening $:
    #   - not escaped
    #   - not part of $$
    #   - not followed by whitespace
    #
    # Closing $:
    #   - not escaped
    #   - not part of $$
    INLINE_MATH_RE =
      /(?<!\\)\$(?!\$|\s)([^$\n]*?\S)(?<!\\)\$(?!\$)/

    def generate(site)
      seen = {}

      site.pages.each do |page|
        process_document(page, seen)
      end

      site.collections.each_value do |collection|
        collection.docs.each do |doc|
          process_document(doc, seen)
        end
      end
    end

    private

    def process_document(doc, seen)
      return if seen[doc.object_id]

      seen[doc.object_id] = true

      return unless doc.respond_to?(:content)
      return if doc.content.nil? || doc.content.empty?

      # Only touch documents that explicitly enable MathJax.
      math_enabled =
        doc.data["math"] == true ||
        doc.data["math"].to_s.downcase == "true"

      return unless math_enabled
      return unless doc.content.include?("$")

      protected_regions = []

      content = doc.content.gsub(PROTECT_RE) do |match|
        protected_regions << match
        "@@MATHPROTECT#{protected_regions.length - 1}@@"
      end

      content = content.gsub(INLINE_MATH_RE) do
        math = Regexp.last_match(1).strip

        # Prevent TeX characters such as <, > and & from being
        # interpreted as HTML before MathJax sees them.
        escaped_math = CGI.escapeHTML(math)

        %(<span class="math-inline" markdown="0">\\(#{escaped_math}\\)</span>)
      end

      protected_regions.each_with_index do |original, index|
        content.gsub!("@@MATHPROTECT#{index}@@") { original }
      end

      doc.content = content
    end
  end
end
