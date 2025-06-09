import math
from tomi import SectionType, SectionNode, BarStepTick


section_patterns = {
    'pattern1': [
        ("Intro", SectionType.Intro, 8),
        ("Verse1", SectionType.Verse, 16),
        ("PreChorus", SectionType.PreChorus, 8),
        ("Chorus1", SectionType.Chorus, 16),
        ("Verse2", SectionType.Verse, 16),
        ("PreChorus", SectionType.PreChorus, 8),
        ("Chorus2", SectionType.Chorus, 16),
        ("Bridge", SectionType.Bridge, 8),
        ("Chorus3", SectionType.Chorus, 16),
        ("Outro", SectionType.Outro, 8)
    ],
    'pattern2': [
        ("Intro", SectionType.Intro, 8),
        ("Verse1", SectionType.Verse, 16),
        ("Chorus1", SectionType.Chorus, 8),
        ("Verse2", SectionType.Verse, 16),
        ("Chorus2", SectionType.Chorus, 8),
        ("Bridge", SectionType.Bridge, 8),
        ("Chorus3", SectionType.Chorus, 8),
        ("Outro", SectionType.Outro, 8)
    ],
    'pattern3': [
        ("Intro", SectionType.Intro, 8),
        ("Verse1", SectionType.Verse, 16),
        ("PreChorus1", SectionType.PreChorus, 8),
        ("Chorus1", SectionType.Chorus, 16),
        ("Verse2", SectionType.Verse, 16),
        ("PreChorus2", SectionType.PreChorus, 8),
        ("Chorus2", SectionType.Chorus, 16),
        ("Bridge", SectionType.Bridge, 8),
        ("Chorus3", SectionType.Chorus, 16),
        ("Outro", SectionType.Outro, 8)
    ],
    'pattern4': [
        ("Intro", SectionType.Intro, 8),
        ("Chorus", SectionType.Chorus, 8),
        ("Verse", SectionType.Verse, 16),
        ("PreChorus", SectionType.PreChorus, 4),
        ("Chorus", SectionType.Chorus, 8),
        ("Verse", SectionType.Verse, 16),
        ("PreChorus", SectionType.PreChorus, 4),
        ("Chorus", SectionType.Chorus, 8),
        ("Outro", SectionType.Outro, 8)
    ]
}


def form_section_prompt(pattern_type: str):
    section_pattern: list[tuple[str, SectionType, int]] = section_patterns[pattern_type]
    sec_strs = [f"{sec[1].name}({sec[2]}bars)" for sec in section_pattern]
    total_bars = sum(sec[2] for sec in section_pattern)
    prompt = f"Given this song structure: [{", ".join(sec_strs)}], the total length is {total_bars} bars."
    return prompt

def structure_and_sections_dict(pattern_type: str) -> dict:
    section_pattern: list[tuple[str, SectionType, int]] = section_patterns[pattern_type]
    structure = [x[0] for x in section_pattern]
    sections = [[x[0], x[1].name, x[2]] for x in section_pattern]
    return {"Structure": structure, "Sections": sections}

def form_structure_and_section_prompt(pattern_type: str):
    section_pattern: list[tuple[str, SectionType, int]] = section_patterns[pattern_type]
    structure = str([x[0] for x in section_pattern]).replace("'", '"')
    sections = str([[x[0], x[1].name, x[2]] for x in section_pattern]).replace("'", '"').replace("[[", "[\n\t\t[").replace("], [", "], \n\t\t[").replace("]]", "]\n\t]")
    prompt = f'Given this song structure and sections:\n{{\n"Structure": {structure},\n"Sections": {sections}\n}}'
    return prompt

def pattern_total_bars(pattern_type: str):
    section_pattern: list[tuple[str, SectionType, int]] = section_patterns[pattern_type]
    total_bars = sum(sec[2] for sec in section_pattern)
    return total_bars

def form_structure(pattern_type: str, project):
    section_pattern: list[tuple[str, SectionType, int]] = section_patterns[pattern_type]
    structure = []
    sec_nodes = {}
    for name, stype, slen in section_pattern:
        if name not in sec_nodes:
            sec_nodes[name] = SectionNode(project, name, stype, BarStepTick(slen))
        structure.append(sec_nodes[name])
    return structure

def get_section_times(pattern_type: str):
    section_pattern: list[tuple[str, SectionType, int]] = section_patterns[pattern_type]
    start = 0
    for name, _, slen in section_pattern:
        time = BarStepTick.bars2sec(slen)
        sminute = str(math.floor(start / 60))
        ssecond = str(int(start % 60))
        end = start + time
        minute = str(math.floor(end / 60))
        second = str(int(end % 60))
        sminute = f"0{sminute}" if len(sminute) == 1 else sminute
        ssecond = f"0{ssecond}" if len(ssecond) == 1 else ssecond
        minute = f"0{minute}" if len(minute) == 1 else minute
        second = f"0{second}" if len(second) == 1 else second
        print(f"{name}: {sminute}:{ssecond} - {minute}:{second}")
        start = end
