#!/usr/bin/env python3
"""Validate just the scheduling section additions"""

import re

def validate_scheduling_section():
    # Read the README
    with open('README.md', 'r') as f:
        content = f.read()

    # Extract the scheduling section
    scheduling_match = re.search(r'## Scheduling Automated Collection(.*?)(?=\n## |\Z)', content, re.DOTALL)
    
    if not scheduling_match:
        print('❌ Scheduling section not found')
        return False
        
    scheduling_content = scheduling_match.group(1)
    print('✅ Scheduling section found')
    
    # Check the exact content matches the specification
    required_elements = [
        '### Setting Up Schedule',
        'ai-news schedule set daily',
        'ai-news schedule set hourly', 
        'ai-news schedule set weekly',
        'ai-news schedule cron-setup',
        'crontab -e',
        '### Managing Schedule',
        'ai-news schedule show',
        'ai-news schedule clear',
        '### Cron Schedule Examples',
        '0 * * * *',
        '0 2 * * *',
        '0 3 * * 0',
        'The tool will collect news from all configured RSS feeds on the specified schedule.'
    ]
    
    all_found = True
    for element in required_elements:
        if element in scheduling_content:
            print(f'✅ Found: {element}')
        else:
            print(f'❌ Missing: {element}')
            all_found = False
            
    # Check the section structure matches exactly what was specified
    print('\n📋 Scheduling section structure validation:')
    
    # Check for proper step-by-step instructions
    if '1. **Configure collection interval:**' in scheduling_content:
        print('✅ Step 1 properly formatted')
    else:
        print('❌ Step 1 formatting issue')
        all_found = False
        
    if '2. **Get cron setup instructions:**' in scheduling_content:
        print('✅ Step 2 properly formatted')
    else:
        print('❌ Step 2 formatting issue')
        all_found = False
        
    if '3. **Add cron job:**' in scheduling_content:
        print('✅ Step 3 properly formatted')
    else:
        print('❌ Step 3 formatting issue')
        all_found = False
    
    # Check for proper code block formatting within the section
    code_blocks_in_section = re.findall(r'```', scheduling_content)
    if len(code_blocks_in_section) % 2 == 0:
        print('✅ Code blocks properly closed in scheduling section')
    else:
        print('❌ Unclosed code block in scheduling section')
        all_found = False
        
    return all_found

if __name__ == '__main__':
    print('🔍 Validating scheduling documentation implementation...\n')
    
    success = validate_scheduling_section()
    
    if success:
        print('\n🎉 All scheduling documentation validation checks passed!')
        print('✅ Task 4 implementation is complete and correct!')
    else:
        print('\n❌ Some validation checks failed')
        exit(1)