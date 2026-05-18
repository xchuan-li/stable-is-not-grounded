"""
generate_benchmark_v3.py

Auto-generates a larger, label-balanced perturbation benchmark (stable_inference_v3).

For each of the five reasoning categories (taxonomic, transitive, causal,
commonsense, contradiction) we define a set of logically grounded seed instances.
Each seed is expanded into a perturbation group with five variants:

    original, paraphrase, lexical_shift, reasoning_path_shift, distractor

Design notes / honest caveats:
- Perturbations are template-driven. This makes generation scalable and the
  semantic-preservation property easy to control, but it also means the
  perturbations are more regular than human-written ones. A model that learns
  the templates could appear artificially stable. This is documented as a
  benchmark limitation; v4 should mix in human-written variants.
- Every category is generated with an exactly balanced number of yes / no
  groups so that per-answer accuracy gaps reflect model bias, not data skew.
- The "no" cases are constructed from genuinely invalid inferences
  (affirming the consequent, denied antecedent, world-knowledge violation,
  mutual exclusivity) so the gold label is defensible.
"""

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "data" / "stable_inference_v3"

SEED = 42
random.seed(SEED)

DISTRACTORS = [
    "The weather today is cloudy.",
    "This topic was discussed last week.",
    "The room temperature is 22 degrees Celsius.",
    "Someone left the door open earlier.",
    "The report is due on Friday.",
    "A blue car is parked outside.",
    "The meeting started at noon.",
    "There are several books on the shelf.",
]


def distractor_sentence():
    return random.choice(DISTRACTORS)


IRREGULAR_PLURAL = {
    "fish": "fish",
    "salmon": "salmon",
}


def pluralize(noun):
    if noun in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[noun]
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def singularize(noun):
    for sg, pl in IRREGULAR_PLURAL.items():
        if noun == pl:
            return sg
    if noun.endswith(("xes", "ches", "shes", "zes")):
        return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


# -----------------------------------------------------------------------------
# Taxonomic:  sub -> mid -> sup
#   yes:  all mid are sup; sub is a mid            => sub is a sup        (valid)
#   no:   all mid are sup; sub is NOT a mid        => sub is a sup?       (no)
# -----------------------------------------------------------------------------

TAXONOMIC_YES = [
    # (sub, sub_syn, mid, mid_syn, sup)
    ("sparrow", "songbird", "bird", "avian", "animal"),
    ("dog", "hound", "mammal", "warm-blooded vertebrate", "animal"),
    ("rose", "garden rose", "flower", "blossom", "plant"),
    ("salmon", "Atlantic salmon", "fish", "aquatic vertebrate", "animal"),
    ("ant", "worker ant", "insect", "arthropod", "animal"),
    ("oak", "oak tree", "tree", "woody plant", "plant"),
    ("Mars", "the red planet", "planet", "celestial body", "object that orbits a star"),
    ("copper", "refined copper", "metal", "metallic element", "chemical element"),
    ("apple", "red apple", "fruit", "edible fruit", "food"),
    ("violin", "concert violin", "string instrument", "bowed instrument", "musical instrument"),
]

TAXONOMIC_NO = [
    # (entity, entity_syn, mid, mid_syn, sup)  -- entity is NOT a mid
    ("bicycle", "mountain bike", "fish", "aquatic animal", "animal"),
    ("table", "wooden table", "reptile", "cold-blooded animal", "animal"),
    ("rock", "granite rock", "bird", "avian", "animal"),
    ("cloud", "rain cloud", "mammal", "vertebrate", "animal"),
    ("spoon", "metal spoon", "insect", "arthropod", "animal"),
    ("lamp", "desk lamp", "tree", "woody plant", "plant"),
    ("number seven", "the integer seven", "flower", "blossom", "plant"),
    ("song", "folk song", "metal", "metallic element", "chemical element"),
    ("shadow", "dark shadow", "fruit", "edible fruit", "food"),
    ("idea", "abstract idea", "planet", "celestial body", "object that orbits a star"),
]


def build_taxonomic(seed, answer, idx):
    sub, sub_syn, mid, mid_syn, sup = seed
    gid = f"tax_v3_{idx:03d}"
    a_sub, a_sup = article(sub), article(sup)
    a_mid, a_subsyn, a_midsyn = article(mid), article(sub_syn), article(mid_syn)
    if answer == "yes":
        original = f"All {pluralize(mid)} are {pluralize(sup)}. {a_sub.capitalize()} {sub} is {a_mid} {mid}. Therefore, is {a_sub} {sub} {a_sup} {sup}?"
        paraphrase = f"Every {mid} belongs to the category of {pluralize(sup)}. {a_sub.capitalize()} {sub} is classified as {a_mid} {mid}. Does this mean {a_sub} {sub} is {a_sup} {sup}?"
        lexical = f"All {pluralize(mid_syn)} are {pluralize(sup)}. {a_subsyn.capitalize()} {sub_syn} is {a_midsyn} {mid_syn}. Therefore, is {a_subsyn} {sub_syn} {a_sup} {sup}?"
        rpath = f"{a_sub.capitalize()} {sub} belongs to the group of {pluralize(mid)}. The group of {pluralize(mid)} falls under {pluralize(sup)}. Based on this, is {a_sub} {sub} {a_sup} {sup}?"
        core = f"{sub}_is_a_{sup}".replace(" ", "_")
    else:
        original = f"All {pluralize(mid)} are {pluralize(sup)}. {a_sub.capitalize()} {sub} is not {a_mid} {mid}. Therefore, is {a_sub} {sub} {a_sup} {sup}?"
        paraphrase = f"Every {mid} belongs to the category of {pluralize(sup)}. {a_sub.capitalize()} {sub} is not {a_mid} {mid}. Does it follow that {a_sub} {sub} is {a_sup} {sup}?"
        lexical = f"All {pluralize(mid_syn)} are {pluralize(sup)}. {a_subsyn.capitalize()} {sub_syn} is not {a_midsyn} {mid_syn}. Therefore, is {a_subsyn} {sub_syn} {a_sup} {sup}?"
        rpath = f"{a_sub.capitalize()} {sub} is not in the group of {pluralize(mid)}. Only {pluralize(mid)} are stated to be {pluralize(sup)} here. Based on this, is {a_sub} {sub} {a_sup} {sup}?"
        core = f"{sub}_is_not_inferable_as_{sup}".replace(" ", "_")
    return assemble(gid, "taxonomic", answer, core, original, paraphrase, lexical, rpath)


# -----------------------------------------------------------------------------
# Transitive:
#   yes:  all A are B; all B are C            => are A C?     (valid)
#   no :  all A are B; X is a B               => is X an A?    (affirming consequent -> no)
# -----------------------------------------------------------------------------

TRANSITIVE_YES = [
    ("roses", "garden roses", "flowers", "blossoms", "plants"),
    ("dogs", "hounds", "mammals", "vertebrates", "animals"),
    ("squares", "regular quadrilaterals", "rectangles", "right-angled quadrilaterals", "polygons"),
    ("sparrows", "songbirds", "birds", "avians", "animals"),
    ("humans", "people", "mammals", "vertebrates", "living organisms"),
    ("oaks", "oak trees", "trees", "woody plants", "plants"),
    ("salmon", "Atlantic salmon", "fish", "aquatic vertebrates", "animals"),
    ("ants", "worker ants", "insects", "arthropods", "animals"),
    ("cars", "automobiles", "vehicles", "motor vehicles", "machines"),
    ("violins", "concert violins", "string instruments", "bowed instruments", "musical instruments"),
]

TRANSITIVE_NO = [
    # all A are B ; X is a B  -> is X an A?  (no)
    ("roses", "garden roses", "flowers", "blossoms", "tulip"),
    ("dogs", "hounds", "mammals", "vertebrates", "cat"),
    ("squares", "regular quadrilaterals", "rectangles", "four-sided shapes", "non-square rectangle"),
    ("sparrows", "songbirds", "birds", "avians", "penguin"),
    ("oaks", "oak trees", "trees", "woody plants", "pine"),
    ("salmon", "Atlantic salmon", "fish", "aquatic vertebrates", "shark"),
    ("ants", "worker ants", "insects", "arthropods", "beetle"),
    ("cars", "automobiles", "vehicles", "motor vehicles", "boat"),
    ("apples", "red apples", "fruits", "edible fruits", "carrot"),
    ("violins", "concert violins", "string instruments", "bowed instruments", "flute"),
]


def build_transitive(seed, answer, idx):
    gid = f"trans_v3_{idx:03d}"
    if answer == "yes":
        a, a_syn, b, b_syn, c = seed
        original = f"All {a} are {b}. All {b} are {c}. Are {a} {c}?"
        paraphrase = f"Every member of {a} is among {b}. Every member of {b} is among {c}. Does it follow that {a} are {c}?"
        lexical = f"All {a_syn} are {b_syn}. All {b_syn} are {c}. Are {a_syn} {c}?"
        rpath = f"{c} include all {b}. {b} include all {a}. Therefore, are {a} {c}?"
        core = f"{a}_are_{c}".replace(" ", "_")
    else:
        a, a_syn, b, b_syn, x = seed
        b_sg, a_sg, b_syn_sg, a_syn_sg = singularize(b), singularize(a), singularize(b_syn), singularize(a_syn)
        ax, ab, aa = article(x), article(b_sg), article(a_sg)
        ab_syn, aa_syn = article(b_syn_sg), article(a_syn_sg)
        original = f"All {a} are {b}. {ax.capitalize()} {x} is {ab} {b_sg}. Is {ax} {x} {aa} {a_sg}?"
        paraphrase = f"Every member of {a} is among {b}. {ax.capitalize()} {x} is {ab} {b_sg}. Does it follow that {ax} {x} is {aa} {a_sg}?"
        lexical = f"All {a_syn} are {b_syn}. {ax.capitalize()} {x} is {ab_syn} {b_syn_sg}. Is {ax} {x} {aa_syn} {a_syn_sg}?"
        rpath = f"Being {ab} {b_sg} does not by itself make something {aa} {a_sg}. {ax.capitalize()} {x} is {ab} {b_sg}. Is {ax} {x} {aa} {a_sg}?"
        core = f"{x}_is_not_inferable_as_{a}".replace(" ", "_")
    return assemble(gid, "transitive", answer, core, original, paraphrase, lexical, rpath)


# -----------------------------------------------------------------------------
# Causal:
#   yes:  if C then E ; C happened          => will E?         (valid)
#   no :  if C then E ; C did NOT happen     => will E?         (no, cannot conclude)
# -----------------------------------------------------------------------------

CAUSAL = [
    # (cause, cause_syn, effect, effect_syn)
    ("the soil is watered", "the soil is irrigated", "the seeds germinate", "the seeds sprout"),
    ("the battery is charged", "the battery is powered up", "the phone turns on", "the phone powers on"),
    ("it rains on the field", "rain falls on the field", "the ground becomes wet", "the ground gets soaked"),
    ("the plant receives sunlight", "the plant gets sunshine", "the plant grows", "the plant develops"),
    ("the heater is switched on", "the heater is activated", "the room becomes warm", "the room heats up"),
    ("the alarm is set", "the alarm is configured", "it rings in the morning", "it sounds in the morning"),
    ("the water reaches 100 degrees Celsius", "the water hits boiling point", "it boils", "it starts boiling"),
    ("the switch is flipped", "the switch is toggled", "the light comes on", "the light illuminates"),
    ("the dough is baked", "the dough is put in the oven", "it becomes bread", "it turns into bread"),
    ("the seed is planted and watered", "the seed is sown and irrigated", "a sprout appears", "a shoot emerges"),
]


def build_causal(seed, answer, idx):
    cause, cause_syn, effect, effect_syn = seed
    gid = f"caus_v3_{idx:03d}"
    if answer == "yes":
        original = f"If {cause}, then {effect}. {cause.capitalize()}. Does it follow that {effect}?"
        paraphrase = f"Whenever {cause}, {effect} as a result. In this case {cause}. Can we conclude that {effect}?"
        lexical = f"If {cause_syn}, then {effect_syn}. {cause_syn.capitalize()}. Does it follow that {effect_syn}?"
        rpath = f"The condition for '{effect}' is met because {cause}. Does it follow that {effect}?"
        core = f"{effect}".replace(" ", "_")
    else:
        original = f"If {cause}, then {effect}. It is not the case that {cause}. Does it follow that {effect}?"
        paraphrase = f"Whenever {cause}, {effect}. However, in this situation it is not true that {cause}. Can we conclude that {effect}?"
        lexical = f"If {cause_syn}, then {effect_syn}. It is not the case that {cause_syn}. Does it follow that {effect_syn}?"
        rpath = f"The only stated cause of '{effect}' is '{cause}'. That cause did not occur. Can we conclude that {effect}?"
        core = f"{effect}_not_inferable_without_cause".replace(" ", "_")
    return assemble(gid, "causal", answer, core, original, paraphrase, lexical, rpath)


# -----------------------------------------------------------------------------
# Commonsense: expected vs unexpected world states
# -----------------------------------------------------------------------------

COMMONSENSE_YES = [
    ("A person drinks water because they are thirsty.", "Someone consumes water because of thirst.",
     "An individual drinks H2O because they feel thirsty.", "Thirst leads people to drink water. A person was thirsty and drank water."),
    ("Ice feels cold to the touch.", "Ice is cold when you touch it.",
     "A block of frozen water feels cold to the touch.", "Frozen water is cold. Ice is frozen water, so it feels cold."),
    ("A fire produces heat.", "Flames give off heat.",
     "A blaze produces heat.", "Combustion releases heat. A fire involves combustion, so it produces heat."),
    ("Birds build nests to raise their young.", "Birds make nests for their offspring.",
     "Avians construct nests to rear their young.", "Raising young needs shelter. Birds build nests as that shelter."),
    ("People sleep at night to rest.", "Humans sleep during the night to recover.",
     "Individuals slumber at night to rest.", "Rest requires sleep. People sleep at night to obtain that rest."),
    ("Rain makes the ground wet.", "Rainfall leaves the ground wet.",
     "Precipitation makes the soil wet.", "Water wets surfaces. Rain is water falling, so it wets the ground."),
    ("A hungry animal looks for food.", "An animal that is hungry searches for food.",
     "A famished creature seeks nourishment.", "Hunger drives food-seeking. The animal is hungry, so it looks for food."),
    ("Wearing a coat keeps you warm in winter.", "A coat keeps a person warm during winter.",
     "Donning a jacket keeps you warm in the cold season.", "Insulation retains body heat. A coat insulates, so it keeps you warm."),
    ("Plants need light to grow.", "Plants require light in order to grow.",
     "Vegetation needs illumination to develop.", "Growth needs energy from light. Plants use light, so they need it to grow."),
    ("A knife is used to cut food.", "People use a knife to cut food.",
     "A blade is used to slice food.", "Cutting needs a sharp edge. A knife has a sharp edge, so it cuts food."),
]

COMMONSENSE_NO = [
    ("A person drinks sand because they are thirsty.", "Someone consumes sand because of thirst.",
     "An individual drinks dry sand because they feel thirsty.", "Thirst is relieved by liquids, not sand. A thirsty person drinking sand does not relieve thirst."),
    ("An ice cube stays frozen inside a burning fire.", "An ice cube remains frozen in a roaring fire.",
     "A block of ice stays solid inside an open flame.", "Fire melts ice. An ice cube in fire would melt, not stay frozen."),
    ("A fire makes nearby objects freeze instantly.", "Flames cause nearby items to freeze immediately.",
     "A blaze freezes surrounding objects at once.", "Fire emits heat, which warms objects. It does not freeze them."),
    ("A fish breathes air comfortably for days on dry land.", "A fish breathes air easily for days out of water.",
     "A fish respires air comfortably for days on dry ground.", "Fish breathe through gills that need water. On land for days a fish cannot breathe."),
    ("A person reads a book in complete darkness with no light.", "Someone reads a book in total darkness without any light.",
     "An individual reads a novel in pitch darkness with no illumination.", "Reading needs light to see text. In complete darkness one cannot read."),
    ("Snow falls upward into the sky from the ground.", "Snow rises from the ground up into the sky.",
     "Snowflakes travel upward from the ground into the air.", "Gravity pulls snow downward. Snow falls down, not up."),
    ("A car runs normally for hours with an empty fuel tank.", "A car drives normally for hours on an empty tank.",
     "An automobile operates for hours with no fuel in the tank.", "Engines need fuel to run. With an empty tank a car cannot run for hours."),
    ("A person stays completely dry while swimming in a pool.", "Someone remains fully dry while swimming in a pool.",
     "An individual stays entirely dry while swimming in water.", "Water wets whatever is submerged. A swimmer in a pool gets wet."),
    ("A plant grows tall in total darkness with no light at all.", "A plant grows tall in complete darkness with no light.",
     "Vegetation grows tall in absolute darkness with no illumination.", "Plants need light for photosynthesis. With no light at all a plant cannot grow tall."),
    ("A heavy rock floats on the surface of calm water.", "A heavy stone floats on top of still water.",
     "A dense boulder floats on the surface of calm water.", "Dense rock sinks in water. A heavy rock does not float."),
]


def build_commonsense(seed, answer, idx):
    original_stmt, para_stmt, lex_stmt, rpath_reason = seed
    gid = f"comm_v3_{idx:03d}"
    q = "Is this expected based on common sense?"
    original = f"{original_stmt} {q}"
    paraphrase = f"{para_stmt} Does this match everyday common sense?"
    lexical = f"{lex_stmt} {q}"
    rpath = f"{rpath_reason} Is the described situation expected based on common sense?"
    core = f"commonsense_{'expected' if answer == 'yes' else 'violation'}_{idx:03d}"
    return assemble(gid, "commonsense", answer, core, original, paraphrase, lexical, rpath)


# -----------------------------------------------------------------------------
# Contradiction: logically consistent (yes) vs self-contradictory (no)
# -----------------------------------------------------------------------------

CONTRADICTION_YES = [
    ("A person can read a book and listen to music at the same time.",
     "Someone is able to read a book while also listening to music.",
     "An individual can peruse a book and hear music simultaneously.",
     "Reading and listening to music are independent activities that can co-occur. So doing both at once is possible."),
    ("A computer can run a program while connected to power.",
     "A computer is able to execute a program while plugged in.",
     "A machine can run software while connected to electricity.",
     "Running a program and being powered are compatible states. So a computer can do both."),
    ("It can be sunny in one city and rainy in another at the same time.",
     "One city can be sunny while a different city is rainy at the same moment.",
     "It can be clear in one town and wet in another simultaneously.",
     "Weather is local. Two cities can have different weather at once, so this is consistent."),
    ("A door can be open in the morning and closed in the evening.",
     "A door is open in the morning and shut by evening.",
     "An entrance can be open in the morning and closed at night.",
     "Different times allow different states. Open then closed at separate times is consistent."),
    ("A student can study mathematics and also play the piano.",
     "A student studies mathematics and additionally plays piano.",
     "A pupil can learn mathematics and also perform on the piano.",
     "Studying math and playing piano are not mutually exclusive. So both can be true."),
    ("A glass can be half full and half empty at the same time.",
     "A glass is simultaneously half full and half empty.",
     "A cup can be half full and half empty at once.",
     "Half full and half empty describe the same state. So this is consistent, not contradictory."),
    ("A train can be moving while a passenger inside is sitting still relative to the train.",
     "A train moves while a passenger inside stays still relative to the carriage.",
     "A locomotive can move while a rider inside is stationary relative to the car.",
     "Motion is relative. The passenger is still relative to the train but moving relative to the ground; no contradiction."),
    ("A number can be both greater than two and less than ten.",
     "A number is simultaneously larger than two and smaller than ten.",
     "A value can be both above two and below ten.",
     "The ranges overlap between two and ten. A number like five satisfies both, so this is consistent."),
    ("A person can be a parent and also a child of someone else.",
     "Someone can be a parent while also being another person's child.",
     "An individual can be a parent and simultaneously someone's offspring.",
     "Being a parent and being a child refer to different relationships. Both can hold at once."),
    ("A light can be off now and on later.",
     "A light is off at present and on at a later time.",
     "A lamp can be off currently and switched on afterward.",
     "Off and on at different times are not contradictory. So this is consistent."),
]

CONTRADICTION_NO = [
    ("A light is completely on and completely off at the same moment in the same way.",
     "A light is fully on and fully off at the same instant in the same sense.",
     "A lamp is entirely on and entirely off at once in the same way.",
     "On and off in the same sense at the same instant are mutually exclusive. They cannot both hold."),
    ("A person is fully asleep and fully awake at the exact same moment.",
     "Someone is completely asleep and completely awake at the same instant.",
     "An individual is entirely asleep and entirely awake at the same time.",
     "Asleep and awake are mutually exclusive states. One cannot be fully both at once."),
    ("A number is both even and odd at the same time.",
     "A single number is simultaneously even and odd.",
     "One integer is both even and odd at once.",
     "Even and odd are mutually exclusive for an integer. A number cannot be both."),
    ("A cup is completely full and completely empty at the same time.",
     "A cup is entirely full and entirely empty simultaneously.",
     "A glass is totally full and totally empty at once.",
     "Completely full and completely empty exclude each other. Both cannot hold at once."),
    ("A triangle has exactly three sides and exactly four sides at once.",
     "A triangle has precisely three and precisely four sides simultaneously.",
     "A three-sided figure has both three and four edges at the same time.",
     "Exactly three and exactly four are incompatible counts. A shape cannot have both."),
    ("The door is fully closed and fully open at the very same instant.",
     "The door is entirely shut and entirely open at the same moment.",
     "The entrance is completely closed and completely open at once.",
     "Fully closed and fully open are opposite states. They cannot coexist at one instant."),
    ("A statement is entirely true and entirely false at the same time in the same sense.",
     "A claim is wholly true and wholly false simultaneously in the same sense.",
     "An assertion is completely true and completely false at once in the same sense.",
     "True and false in the same sense are mutually exclusive. A statement cannot be both."),
    ("A person is taller than themselves.",
     "Someone is taller than their own height.",
     "An individual exceeds their own height.",
     "Nothing can be taller than itself. This is self-contradictory."),
    ("A box is inside itself completely.",
     "A box is fully contained within its own self.",
     "A container is entirely inside itself.",
     "An object cannot completely contain itself. This is impossible."),
    ("It is raining everywhere and raining nowhere at the same time.",
     "It rains in all places and in no places simultaneously.",
     "Rain falls everywhere and nowhere at the same moment.",
     "Everywhere and nowhere are complete opposites. Both cannot be true at once."),
]


def build_contradiction(seed, answer, idx):
    original_stmt, para_stmt, lex_stmt, rpath_reason = seed
    gid = f"cont_v3_{idx:03d}"
    q = "Is this logically consistent?"
    original = f"{original_stmt} {q}"
    paraphrase = f"{para_stmt} Does this make logical sense?"
    lexical = f"{lex_stmt} {q}"
    rpath = f"{rpath_reason} Is the described situation logically consistent?"
    core = f"contradiction_{'consistent' if answer == 'yes' else 'inconsistent'}_{idx:03d}"
    return assemble(gid, "contradiction", answer, core, original, paraphrase, lexical, rpath)


# -----------------------------------------------------------------------------
# Shared assembly
# -----------------------------------------------------------------------------

def assemble(gid, reasoning_type, answer, core, original, paraphrase, lexical, rpath):
    samples = [
        ("original", original),
        ("paraphrase", paraphrase),
        ("lexical_shift", lexical),
        ("reasoning_path_shift", rpath),
        ("distractor", append_distractor(original)),
    ]
    return {
        "group_id": gid,
        "reasoning_type": reasoning_type,
        "expected_answer": answer,
        "core_inference": core,
        "difficulty": "easy",
        "semantic_risk": "low",
        "source_type": "auto_v3_template",
        "samples": [
            {
                "sample_id": f"{gid}_{ptype}",
                "type": ptype,
                "text": text,
                "answer": answer,
                "validity_label": "valid",
            }
            for ptype, text in samples
        ],
    }


def append_distractor(text):
    # Insert an irrelevant sentence before the final question.
    if "?" in text:
        head, _, tail = text.rpartition(". ")
        if head:
            return f"{head}. {distractor_sentence()} {tail}"
    return f"{text} {distractor_sentence()}"


# -----------------------------------------------------------------------------
# Generation driver
# -----------------------------------------------------------------------------

CATEGORIES = [
    ("taxonomic", TAXONOMIC_YES, TAXONOMIC_NO, build_taxonomic),
    ("transitive", TRANSITIVE_YES, TRANSITIVE_NO, build_transitive),
    ("causal", CAUSAL, CAUSAL, build_causal),
    ("commonsense", COMMONSENSE_YES, COMMONSENSE_NO, build_commonsense),
    ("contradiction", CONTRADICTION_YES, CONTRADICTION_NO, build_contradiction),
]


def main():
    if OUT_DIR.exists():
        for p in sorted(OUT_DIR.rglob("*.json")):
            p.unlink()
    total = 0
    summary = {}
    for name, yes_seeds, no_seeds, builder in CATEGORIES:
        cat_dir = OUT_DIR / name
        cat_dir.mkdir(parents=True, exist_ok=True)
        idx = 1
        yes_count = no_count = 0
        for seed in yes_seeds:
            group = builder(seed, "yes", idx)
            (cat_dir / f"{group['group_id']}.json").write_text(
                json.dumps(group, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            idx += 1
            yes_count += 1
            total += 1
        for seed in no_seeds:
            group = builder(seed, "no", idx)
            (cat_dir / f"{group['group_id']}.json").write_text(
                json.dumps(group, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            idx += 1
            no_count += 1
            total += 1
        summary[name] = (yes_count, no_count)

    print(f"Generated {total} perturbation groups into: {OUT_DIR}")
    print(f"{'Category':<16} {'yes':>5} {'no':>5} {'total':>6}")
    print("-" * 36)
    for name, (y, n) in summary.items():
        print(f"{name:<16} {y:>5} {n:>5} {y + n:>6}")
    print(f"\nTotal samples: {total * 5} (each group has 5 perturbation variants)")


if __name__ == "__main__":
    main()
