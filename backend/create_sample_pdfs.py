"""
Script to create sample therapeutic PDF documents for the knowledge base.
These PDFs contain evidence-based content about digital wellness and screen time management.
"""

from fpdf import FPDF
import os

class TherapeuticPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Digital Wellness Therapy Guide', 0, 1, 'C')
        self.ln(5)
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

def create_screen_time_addiction_pdf():
    """Create PDF about screen time addiction and digital dependency."""
    pdf = TherapeuticPDF()
    pdf.add_page()
    
    pdf.chapter_title("Understanding Screen Time Addiction")
    pdf.chapter_body("""
Screen time addiction, also known as digital dependency, is characterized by compulsive use of digital devices 
that interferes with daily life, relationships, and well-being. It's not about the amount of time spent online, 
but rather the inability to control device use despite negative consequences.

Common Signs:
- Feeling anxious or irritable when unable to access devices
- Neglecting responsibilities, relationships, or self-care
- Using devices to escape negative emotions
- Difficulty sleeping due to late-night device use
- Physical symptoms like eye strain, headaches, or neck pain
- Loss of interest in offline activities once enjoyed
""")
    
    pdf.chapter_title("The Psychology Behind Digital Dependency")
    pdf.chapter_body("""
Digital devices and social media platforms are designed to be addictive. They exploit psychological principles:

1. Variable Rewards: Social media notifications trigger dopamine release, similar to slot machines
2. FOMO (Fear of Missing Out): Creates anxiety about being disconnected
3. Social Validation: Likes and comments provide instant gratification
4. Infinite Scroll: Removes natural stopping cues
5. Emotional Regulation: Devices become tools for managing uncomfortable feelings

Understanding these mechanisms helps us recognize we're not weak-willed; we're dealing with sophisticated 
behavioral engineering designed to capture our attention.
""")
    
    pdf.chapter_title("Cognitive Behavioral Therapy (CBT) Approaches")
    pdf.chapter_body("""
CBT helps identify and change thought patterns that drive excessive screen time:

Identifying Triggers:
- What emotions precede device use? (boredom, loneliness, stress)
- What situations trigger automatic reaching for your phone?
- What thoughts justify excessive use?

Challenging Automatic Thoughts:
- "I need to check my phone now" becomes "I can wait 10 minutes"
- "Everyone else is online" becomes "I choose how I spend my time"
- "This is relaxing" becomes "Is this actually making me feel better?"

Behavioral Experiments:
- Try a 1-hour phone-free period and observe your feelings
- Notice what happens when you don't check notifications immediately
- Experiment with device-free meals or conversations
""")
    
    pdf.chapter_title("Mindfulness and Present-Moment Awareness")
    pdf.chapter_body("""
Mindfulness practices help create space between impulse and action:

STOP Technique:
S - Stop what you're doing
T - Take a breath
O - Observe your thoughts, feelings, and urges
P - Proceed with awareness

Urge Surfing:
When you feel the urge to check your device, imagine the urge as a wave. Rather than immediately acting on it, 
notice it rise, peak, and naturally subside. Urges are temporary and will pass without action.

Mindful Device Use:
Before picking up your device, pause and ask:
- What am I hoping to accomplish?
- How do I want to feel afterward?
- Is this aligned with my values?
""")
    
    pdf.chapter_title("Acceptance and Commitment Therapy (ACT) Strategies")
    pdf.chapter_body("""
ACT focuses on clarifying values and committed action:

Values Clarification:
What matters most to you? Consider:
- Relationships: How do you want to show up for loved ones?
- Health: What does physical and mental wellness mean to you?
- Growth: What skills or experiences do you want to pursue?
- Contribution: How do you want to impact your community?

Committed Action:
Align daily choices with values:
- If family is important, commit to device-free dinners
- If health matters, replace evening scrolling with movement
- If creativity is valued, dedicate reclaimed time to creative pursuits

Psychological Flexibility:
Notice thoughts about device use without judgment. You can have the thought "I want to check my phone" without 
acting on it. Thoughts are mental events, not commands.
""")
    
    pdf.add_page()
    pdf.chapter_title("Practical Boundary-Setting Strategies")
    pdf.chapter_body("""
Effective boundaries require both environmental design and internal commitment:

Time-Based Boundaries:
- Device-free first hour after waking
- No screens 1 hour before bed
- Designated "phone-free" times during meals
- Technology sabbaths (one day per week)

Space-Based Boundaries:
- Keep phones out of bedroom
- Charge devices away from reach at night
- Create phone-free zones (dining room, bathroom)
- Use physical timers instead of phone alarms

App-Based Boundaries:
- Turn off non-essential notifications
- Use grayscale mode to reduce appeal
- Set app time limits
- Uninstall problematic apps
- Use website blockers during work hours

Social Boundaries:
- Communicate your boundaries to friends and family
- Set expectations about response times
- Use auto-replies during focus periods
- Practice saying "I'm taking a digital break"
""")
    
    pdf.chapter_title("Motivational Interviewing Approach")
    pdf.chapter_body("""
Change begins with exploring ambivalence:

Exploring Discrepancy:
What do you value? How does current device use align with those values?
What would your ideal relationship with technology look like?

Scaling Questions:
On a scale of 1-10:
- How important is reducing screen time to you?
- How confident are you in your ability to change?

Change Talk:
Listen for your own reasons to change:
- Desire: "I want more time with my family"
- Ability: "I was able to go phone-free during that vacation"
- Reasons: "I sleep better when I don't use screens before bed"
- Need: "I need to be more present in my life"

Small Steps:
What's one small change you could make this week?
Success builds confidence for larger changes.
""")
    
    pdf.chapter_title("Managing Emotional Triggers")
    pdf.chapter_body("""
Many people use devices to manage difficult emotions:

Emotional Awareness:
Before reaching for your device, pause and identify:
- What am I feeling right now?
- What do I need in this moment?
- Will device use address this need?

Alternative Coping Strategies:
For each emotional trigger, identify healthier responses:

Boredom:
- Call a friend
- Go for a walk
- Engage in a hobby
- Do a creative activity

Loneliness:
- Reach out to someone directly (call, don't text)
- Join a community activity
- Practice self-compassion
- Connect with nature

Stress:
- Practice deep breathing
- Physical exercise
- Journal
- Listen to music

Anxiety:
- Progressive muscle relaxation
- Grounding techniques (5-4-3-2-1)
- Talk to someone
- Engage in problem-solving

The goal isn't to never use devices, but to have multiple tools for emotional regulation.
""")
    
    pdf.add_page()
    pdf.chapter_title("Building a Supportive Environment")
    pdf.chapter_body("""
Sustainable change requires environmental support:

Social Support:
- Find an accountability partner
- Join digital wellness groups
- Share your goals with family and friends
- Model healthy device use for children

Identity Shift:
Rather than "I'm trying to use my phone less," consider:
"I'm someone who values presence and connection"
"I'm someone who protects my attention"
"I'm someone who uses technology intentionally"

Celebrating Progress:
Notice and celebrate small wins:
- Completed a meal without checking phone
- Woke up without immediately scrolling
- Had a meaningful conversation without distractions
- Chose an offline activity over device use

Self-Compassion:
Lapses are normal. Rather than self-criticism:
- Notice what triggered the lapse
- Recommit to your values
- Adjust strategies if needed
- Remember: progress, not perfection
""")
    
    pdf.chapter_title("Crisis Support and When to Seek Professional Help")
    pdf.chapter_body("""
While digital wellness is important, some situations require professional support:

Seek immediate help if you experience:
- Suicidal thoughts or self-harm urges
- Severe depression or anxiety
- Complete withdrawal from daily activities
- Relationship or job loss due to device use

Resources:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- SAMHSA National Helpline: 1-800-662-4357

Consider professional therapy if:
- Self-help strategies aren't working
- Device use significantly impacts functioning
- Underlying mental health concerns exist
- You need more structured support

A therapist can provide:
- Personalized assessment and treatment
- Evidence-based interventions
- Underlying issue exploration
- Medication evaluation if needed
- Ongoing support and accountability
""")
    
    os.makedirs('knowledge_base', exist_ok=True)
    pdf.output('knowledge_base/screen_time_addiction_guide.pdf')
    print("Created: screen_time_addiction_guide.pdf")


def create_social_media_boundaries_pdf():
    """Create PDF about social media and technology boundaries."""
    pdf = TherapeuticPDF()
    pdf.add_page()
    
    pdf.chapter_title("Understanding Social Media Impact on Mental Health")
    pdf.chapter_body("""
Social media has transformed how we connect, but research reveals significant mental health impacts:

Documented Effects:
- Increased rates of anxiety and depression, especially in young adults
- Social comparison leading to lower self-esteem
- FOMO (Fear of Missing Out) creating constant anxiety
- Disrupted sleep patterns from evening use
- Reduced attention span and concentration
- Decreased face-to-face social skills

However, these effects aren't inevitable. Mindful, intentional use can maintain benefits while minimizing harm.
""")
    
    pdf.chapter_title("The Comparison Trap")
    pdf.chapter_body("""
Social media presents highlight reels, not reality:

Understanding Comparison:
- We compare our behind-the-scenes with others' highlight reels
- Upward comparison (to those seemingly better off) increases dissatisfaction
- Social media amplifies natural comparison tendencies

Cognitive Strategies:
1. Reality Testing: "What am I not seeing in this post?"
2. Gratitude Practice: Focus on your own blessings
3. Compassion: Remember everyone struggles, even those with perfect feeds
4. Limiting Exposure: Unfollow accounts that trigger comparison

Reframing Thoughts:
Instead of: "Everyone else has a perfect life"
Try: "People share their best moments, not their struggles"

Instead of: "I'm not good enough"
Try: "I'm on my own unique journey"
""")
    
    pdf.chapter_title("Setting Healthy Social Media Boundaries")
    pdf.chapter_body("""
Boundaries protect your mental health and time:

Consumption Boundaries:
- Limit daily usage (start with 30-60 minutes)
- Use app timers and tracking tools
- Schedule specific social media times
- Avoid morning and bedtime scrolling
- Take regular social media breaks (days or weeks)

Content Boundaries:
- Curate your feed intentionally
- Unfollow accounts that trigger negative feelings
- Follow accounts aligned with your values
- Mute or block toxic individuals
- Join positive, supportive communities

Engagement Boundaries:
- Don't feel obligated to respond immediately
- Set limits on comment reading
- Avoid late-night debates or discussions
- Practice digital detachment from others' drama
- Remember: you don't owe anyone your attention

Creation Boundaries:
- Post intentionally, not compulsively
- Don't seek validation through likes/comments
- Share authentically, not performatively
- Protect your privacy (limit personal information)
- Consider if posts align with your values
""")
    
    pdf.chapter_title("Reclaiming Your Attention")
    pdf.chapter_body("""
Attention is our most valuable resource:

Understanding Attention Economy:
Social media platforms profit from your attention. Every feature is designed to maximize engagement:
- Infinite scroll removes stopping points
- Autoplay keeps you watching
- Notifications interrupt focus
- Algorithms show emotionally triggering content
- Variable rewards create addictive patterns

Strategies for Protection:
1. Turn Off Notifications: Check apps intentionally, not reactively
2. Remove Apps from Home Screen: Add friction to access
3. Use Website Versions: Less engaging than apps
4. Batch Check Times: Reduce context-switching
5. Use Focus Modes: During work, sleep, family time

Deep Work Practice:
- Schedule uninterrupted work blocks
- Remove all digital distractions
- Practice single-tasking
- Build tolerance for boredom
- Strengthen attention muscle through practice

Mindful Consumption:
Before opening social media, ask:
- What's my intention?
- How much time will I spend?
- How do I want to feel afterward?
Set a timer and stick to it.
""")
    
    pdf.add_page()
    pdf.chapter_title("Alternative Connection Strategies")
    pdf.chapter_body("""
Reduce social media while maintaining genuine connection:

Quality Over Quantity:
- Prioritize deep conversations over likes
- Call friends instead of commenting
- Meet in person when possible
- Join local groups or classes
- Volunteer in your community

Meaningful Digital Connection:
- Send personal messages, not public posts
- Share authentically in small groups
- Use video calls for distant loved ones
- Write emails or letters
- Join interest-based online communities

Offline Relationship Building:
- Schedule regular friend dates
- Join clubs aligned with interests
- Attend community events
- Practice hobbies in group settings
- Participate in local sports or activities

The goal isn't isolation, but intentional connection that nourishes rather than depletes.
""")
    
    pdf.chapter_title("Identity Beyond the Feed")
    pdf.chapter_body("""
Social media can distort our sense of self:

Authentic Self vs. Online Persona:
Many people curate idealized versions online:
- Only sharing positive moments
- Seeking external validation
- Performing for an audience
- Losing touch with genuine preferences

Reconnecting with Authentic Self:
Ask yourself:
- Who am I when no one is watching?
- What do I enjoy without sharing it?
- What are my values independent of others' opinions?
- What activities bring me joy regardless of "shareability"?

Building Offline Identity:
- Engage in activities you won't post about
- Develop skills for personal satisfaction
- Create private memories
- Value internal validation over likes
- Define success by your own standards

Remember: Your worth isn't measured in followers, likes, or shares. You are valuable simply because you exist.
""")
    
    pdf.chapter_title("Digital Minimalism Philosophy")
    pdf.chapter_body("""
Digital minimalism: intentionally using technology in ways that support your values.

Core Principles:
1. Intentionality: Every app and platform serves a clear purpose
2. Optimization: Use technology in ways that maximize value
3. Selectivity: Say no to most technologies, yes to few
4. Regular Review: Continuously evaluate what stays and goes

Implementation:
1. Digital Declutter (30 days):
   - Remove all optional technologies
   - Identify what you genuinely missed
   - Reintroduce selectively with rules

2. Operating Procedures:
   For each technology you reintroduce, define:
   - When: Specific times for use
   - Where: Specific locations allowed
   - How: Clear guidelines for use

3. Analog Alternatives:
   - Physical books instead of e-readers
   - Paper planners over digital calendars
   - Real alarm clocks, not phone alarms
   - Print maps occasionally
   - Handwritten notes

Benefits:
- Increased focus and productivity
- Improved relationships
- Greater life satisfaction
- More time for meaningful activities
- Reduced anxiety and stress
""")
    
    os.makedirs('knowledge_base', exist_ok=True)
    pdf.output('knowledge_base/social_media_boundaries_guide.pdf')
    print("Created: social_media_boundaries_guide.pdf")


def create_therapeutic_techniques_pdf():
    """Create PDF about therapeutic techniques and approaches."""
    pdf = TherapeuticPDF()
    pdf.add_page()
    
    pdf.chapter_title("Evidence-Based Therapeutic Approaches for Digital Wellness")
    pdf.chapter_body("""
This guide integrates multiple therapeutic modalities to address digital dependency:

1. Cognitive Behavioral Therapy (CBT)
2. Mindfulness-Based Interventions
3. Acceptance and Commitment Therapy (ACT)
4. Motivational Interviewing
5. Dialectical Behavior Therapy (DBT) skills

Each approach offers unique tools for managing technology use and improving well-being.
""")
    
    pdf.chapter_title("CBT: Thought Records for Device Use")
    pdf.chapter_body("""
Track and challenge thoughts that drive excessive device use:

Situation: Describe when/where urge occurs
Automatic Thought: What went through your mind?
Emotion: What did you feel? (rate 0-100)
Behavior: What did you do?
Alternative Thought: More balanced perspective
Outcome: How do you feel now? (rate 0-100)

Example:
Situation: Home alone on Friday evening
Automatic Thought: "I have nothing better to do than scroll"
Emotion: Boredom (70), Loneliness (60)
Behavior: Scrolled social media for 3 hours
Alternative Thought: "I could call a friend, read, or work on my art project"
Outcome: Boredom (50), Loneliness (40), Regret (70)

Over time, this practice:
- Increases awareness of patterns
- Challenges automatic thoughts
- Builds healthier response options
- Reduces impulsive device use
""")
    
    pdf.chapter_title("Mindfulness: RAIN Technique")
    pdf.chapter_body("""
RAIN is a four-step mindfulness practice for working with difficult urges:

R - Recognize:
Notice when you feel the urge to use your device. Simply acknowledge: "I'm feeling the urge to check my phone."

A - Allow:
Let the feeling be present without trying to change it. The urge doesn't require action. It's just a sensation.

I - Investigate:
With curiosity, explore the urge:
- Where do you feel it in your body?
- What emotion is underneath?
- What need is present?
- What's the quality of the sensation?

N - Nurture:
Offer yourself compassion. Place a hand on your heart and say: "This is hard. I'm doing my best. May I be kind to myself."

After RAIN, you can choose how to respond. Often, the urge has passed or decreased significantly.
""")
    
    pdf.chapter_title("ACT: Values-Based Goal Setting")
    pdf.chapter_body("""
Align technology use with your deepest values:

Step 1: Identify Core Values
In each life area, what matters most to you?

Relationships:
- Being present with loved ones
- Building deep connections
- Being a good listener
- Showing up consistently

Personal Growth:
- Learning new skills
- Pursuing creative interests
- Physical health
- Emotional well-being

Contribution:
- Helping others
- Making a difference
- Being part of community
- Using talents meaningfully

Step 2: Rate Current Alignment
For each value, rate 1-10: How well do current technology habits support this value?

Step 3: Set Valued Goals
Choose one value. Create a SMART goal:
- Specific: Exactly what will you do?
- Measurable: How will you track it?
- Achievable: Is it realistic?
- Relevant: Does it align with the value?
- Time-bound: When will you do this?

Example:
Value: Being present with family
Current Rating: 4/10
Goal: Have device-free dinners Monday-Friday for the next month

Step 4: Identify Barriers
What obstacles might arise? Plan strategies for each.

Step 5: Track and Adjust
Review progress weekly. Celebrate successes. Adjust as needed.
""")
    
    pdf.add_page()
    pdf.chapter_title("Motivational Interviewing: Change Talk")
    pdf.chapter_body("""
Strengthen your own motivation by articulating reasons for change:

DARN-CAT Framework:

Desire:
- "I want to feel more present"
- "I want better sleep"
- "I want deeper relationships"

Ability:
- "I was able to leave my phone home for a hike"
- "I can focus without my phone for 30 minutes"
- "I know how to turn off notifications"

Reasons:
- "I sleep better without screens before bed"
- "My relationships improve when I'm not distracted"
- "I accomplish more without constant interruptions"

Need:
- "I need to be more present for my kids"
- "I need to protect my mental health"
- "I need to reclaim my time"

Commitment:
- "I will put my phone in another room at night"
- "I'm committed to device-free meals"

Activation:
- "I'm ready to make changes"
- "I'm willing to try new strategies"

Taking Steps:
- "I've already turned off social media notifications"
- "I deleted Instagram from my phone yesterday"

Practice saying these statements aloud. Hearing yourself express motivation strengthens commitment.
""")
    
    pdf.chapter_title("DBT: Distress Tolerance Skills")
    pdf.chapter_body("""
Manage urges without giving in using DBT TIPP skills:

T - Temperature:
Change your body temperature to shift emotional state:
- Splash cold water on your face
- Hold ice cubes
- Take a cold shower
This triggers the dive reflex, calming your nervous system.

I - Intense Exercise:
Do brief, intense physical activity:
- 50 jumping jacks
- Sprint up stairs
- Push-ups or squats
- Dance energetically
Burns off adrenaline and shifts focus.

P - Paced Breathing:
Slow your breathing to calm anxiety:
- Breathe in for 4 counts
- Hold for 4 counts
- Breathe out for 6 counts
- Repeat for 5 minutes

P - Paired Muscle Relaxation:
Combine breathing with muscle relaxation:
- Tense muscles while breathing in
- Release while breathing out
- Move through body: feet to head

When urges feel overwhelming, use TIPP skills to get through the moment without acting impulsively.
""")
    
    pdf.chapter_title("Building Sustainable Habits")
    pdf.chapter_body("""
Transform insights into lasting change:

Start Small:
- Choose one behavior to change
- Make it ridiculously easy
- Build confidence through success
- Gradually increase difficulty

Habit Stacking:
Link new habits to existing ones:
"After I [existing habit], I will [new habit]"

Examples:
- "After I brush my teeth, I'll put my phone on the charger in another room"
- "After I eat dinner, I'll go for a 10-minute walk without my phone"
- "After I wake up, I'll meditate for 5 minutes before checking my phone"

Implementation Intentions:
Specify when, where, and how you'll act:

"If [situation], then I will [behavior]"

Examples:
- "If I feel the urge to check social media, then I will take three deep breaths"
- "If I'm waiting in line, then I will notice my surroundings instead of checking my phone"
- "If I wake up during the night, then I will do a body scan instead of scrolling"

Environment Design:
- Make healthy choices easy
- Make unhealthy choices hard
- Use physical reminders
- Create accountability systems

Progress Tracking:
- Keep a daily log
- Celebrate small wins
- Learn from lapses
- Adjust strategies
- Share progress with others
""")
    
    pdf.chapter_title("Self-Compassion in Behavior Change")
    pdf.chapter_body("""
Treat yourself with kindness during the change process:

Three Components of Self-Compassion:

1. Self-Kindness vs. Self-Judgment:
When you slip up, speak to yourself as you would a good friend:

Instead of: "I'm so weak, I can't even stay off my phone"
Try: "This is challenging. I'm learning and growing. Tomorrow is a new day."

2. Common Humanity vs. Isolation:
Remember that struggle is part of being human:

Instead of: "I'm the only one who can't control device use"
Try: "Millions of people struggle with this. I'm not alone. This is a widespread challenge."

3. Mindfulness vs. Over-Identification:
Observe your experience without getting swept away:

Instead of: "I'm a failure, I'll never change"
Try: "I'm noticing thoughts of failure. I'm experiencing difficulty. This is temporary."

Self-Compassion Break:
When struggling, place hand on heart and say:
1. "This is a moment of suffering" (mindfulness)
2. "Suffering is part of life" (common humanity)
3. "May I be kind to myself" (self-kindness)

Research shows self-compassion is more effective than self-criticism for behavior change. Kindness motivates; criticism depletes.
""")
    
    os.makedirs('knowledge_base', exist_ok=True)
    pdf.output('knowledge_base/therapeutic_techniques_guide.pdf')
    print("Created: therapeutic_techniques_guide.pdf")


if __name__ == "__main__":
    print("Creating sample therapeutic PDFs for knowledge base...")
    print("-" * 60)
    
    create_screen_time_addiction_pdf()
    create_social_media_boundaries_pdf()
    create_therapeutic_techniques_pdf()
    
    print("-" * 60)
    print("All PDFs created successfully in knowledge_base/ directory!")
    print("These PDFs contain evidence-based therapeutic content for the RAG system.")
