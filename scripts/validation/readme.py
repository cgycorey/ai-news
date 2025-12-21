#!/usr/bin/env python3
"""Validate README.md markdown syntax and content"""

import re
import sys

def main():
    # Read the README
    with open('README.md', 'r') as f:
        content = f.read()

    issues = []

    # Check for proper heading spacing (## followed by space)
    if re.search(r'^##[^\s]', content, re.MULTILINE):
        issues.append('Found headings without space after ##')

    # Check for unclosed code blocks
    code_blocks = re.findall(r'```', content)
    if len(code_blocks) % 2 != 0:
        issues.append('Unclosed code block detected')

    # Check that our section was added
    if '## Scheduling Automated Collection' not in content:
        issues.append('Scheduling section not found')
    else:
        print('✅ Scheduling section found')

    # Check for required subsections
    required_sections = [
        '### Setting Up Schedule',
        '### Managing Schedule', 
        '### Cron Schedule Examples'
    ]

    for section in required_sections:
        if section in content:
            print(f'✅ {section} found')
        else:
            issues.append(f'Missing section: {section}')

    # Check for key commands
    key_commands = [
        'ai-news schedule set daily',
        'ai-news schedule set hourly',
        'ai-news schedule set weekly',
        'ai-news schedule cron-setup',
        'ai-news schedule show',
        'ai-news schedule clear'
    ]

    for cmd in key_commands:
        if cmd in content:
            print(f'✅ Command found: {cmd}')
        else:
            issues.append(f'Missing command: {cmd}')

    # Check cron examples
    cron_examples = ['0 * * * *', '0 2 * * *', '0 3 * * 0']
    for cron in cron_examples:
        if cron in content:
            print(f'✅ Cron example found: {cron}')
        else:
            issues.append(f'Missing cron example: {cron}')

    if issues:
        print('\n❌ Issues found:')
        for issue in issues:
            print(f'  - {issue}')
        sys.exit(1)
    else:
        print('\n✅ All markdown validation checks passed!')
        print('✅ Scheduling documentation implemented successfully!')

if __name__ == '__main__':
    main()