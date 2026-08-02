from unittest.mock import MagicMock


def test_assembled_context_prompt_block_is_compact():
    from book_graph_analyzer.generate.context import AssembledContext
    from book_graph_analyzer.generate.shadow.models import CharacterState

    context = AssembledContext(
        character_states=[
            CharacterState(
                name="Tuor",
                story_id="story-1",
                location="Echoriath foothills",
                possessions=["Sword of Turgon", "Grey Cloak"],
                conditions=["weary", "determined"],
            ),
            CharacterState(
                name="Voronwe",
                story_id="story-1",
                location="Echoriath foothills",
                conditions=["resolute"],
            ),
        ],
        recent_summaries=[
            "Tuor and Voronwe fled an Orc ambush near Sirion.",
            "They found a hidden pass marked by Ulmo's sign.",
            "Voronwe revealed the Hidden City's name.",
        ],
        place_facts={
            "name": "Echoriath",
            "region": "Beleriand",
            "description": "The Encircling Mountains around Gondolin.",
            "facts": ["Hidden passes are guarded", "The ridges are steep and cold"],
        },
        active_plot_threads=[
            "Reach Gondolin without alerting Morgoth's scouts",
            "Protect Tuor's mission from Ulmo",
        ],
    )

    block = context.to_prompt_block()
    assert "CURRENT STATE" in block
    assert "RECENT EVENTS" in block
    assert "CURRENT PLACE" in block
    assert "ACTIVE PLOT THREADS" in block

    estimated_tokens = len(block) / 4
    assert estimated_tokens < 400


def test_context_assembler_merges_shadow_and_neo4j_data():
    from book_graph_analyzer.generate.context import ContextAssembler
    from book_graph_analyzer.generate.shadow.models import CharacterState, SceneState

    shadow_graph = MagicMock()
    shadow_graph.story_id = "story-1"
    shadow_graph.get_scene_state.return_value = SceneState(
        characters=[CharacterState(name="Tuor", story_id="story-1", location="Echoriath")],
        recent_summaries=["They crossed a frozen stream."],
    )

    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False

    place_result = MagicMock()
    place_result.single.return_value = {
        "name": "Echoriath",
        "description": "A wall of mountains guarding Gondolin.",
        "region": "Beleriand",
        "type": "mountains",
        "history": "Known to hide secret ways.",
    }

    outline_rows = [
        {
            "story_outline": "- Reach Gondolin\n- Avoid Orc patrols",
            "chapter_num": 1,
            "chapter_outline": "- Cross the foothills",
        }
    ]

    def run_side_effect(query, **kwargs):
        if "MATCH (p:Place)" in query:
            return place_result
        result = MagicMock()
        result.__iter__.return_value = iter(outline_rows)
        return result

    session.run.side_effect = run_side_effect

    assembler = ContextAssembler(shadow_graph=shadow_graph, neo4j_driver=driver)
    assembled = assembler.assemble(
        story_id="story-1",
        characters=["Tuor"],
        place="Echoriath",
        chapter_num=1,
        scene_num=2,
    )

    assert assembled.character_states[0].name == "Tuor"
    assert assembled.place_facts["name"] == "Echoriath"
    assert any("Reach Gondolin" in t for t in assembled.active_plot_threads)


def test_scene_generator_accepts_assembled_context():
    from book_graph_analyzer.generate.context import AssembledContext
    from book_graph_analyzer.generate.generator import SceneGenerator

    generator = SceneGenerator()
    generator.driver = None
    generator.llm = MagicMock()
    generator.llm.generate.return_value = "Scene text"
    generator.llm.provider_label = "stub:test-model"
    from book_graph_analyzer.generate.models import SceneScores

    generator.judge = MagicMock()
    scores = SceneScores(style_score=0.8, narrative_score=0.8)
    generator.judge.full_evaluation.return_value = (scores, "", [])
    generator._critique_scene = MagicMock(return_value=[])

    assembled = AssembledContext(
        recent_summaries=["Summary A"],
        place_facts={"name": "Gondolin", "description": "Hidden city."},
    )

    scene = generator.generate_scene(
        scene_goal="Advance toward the hidden gate",
        characters=["Tuor"],
        place="Gondolin approaches",
        assembled_context=assembled,
    )

    assert scene.context_snapshot is assembled
    assert "CURRENT STATE" in scene.generation_prompt
    assert scene.scene_type == "discovery"
    assert "STYLE GUIDE (fallback):" in scene.generation_prompt
    assert scene.style_constraints_used is None
    assert scene.model_used == "stub:test-model"


def test_scene_generator_prompts_use_requested_length_and_keep_default_range():
    from book_graph_analyzer.generate.generator import SceneGenerator
    from book_graph_analyzer.generate.models import SceneScores

    generator = SceneGenerator()
    generator.driver = None
    generator.llm = MagicMock()
    generator.llm.generate.return_value = "Scene text"
    generator.llm.provider_label = "stub:test-model"
    generator.pipeline.run = MagicMock(side_effect=lambda scene, **_kwargs: (scene, []))
    generator._score_scene = MagicMock(return_value=SceneScores())

    normal_scene = generator.generate_scene(
        scene_goal="Follow the trail east",
        characters=["Aragorn"],
        place="Wilderland",
        target_words=1050,
    )
    fog_scene = generator.generate_scene(
        scene_goal="Enter the lightless ruin",
        characters=["Aragorn"],
        place="an unnamed ruin",
        fog_of_war=True,
        target_words=900,
    )
    default_scene = generator.generate_scene(
        scene_goal="Keep watch",
        characters=["Aragorn"],
        place="the road",
    )

    assert (
        "Write approximately 1050 words (within 10 percent of the target)."
        in normal_scene.generation_prompt
    )
    assert (
        "Write approximately 900 words (within 10 percent of the target)."
        in fog_scene.generation_prompt
    )
    assert "Write 400-800 words." in default_scene.generation_prompt
    assert "avoid modern analytical, clinical, or process jargon" in normal_scene.generation_prompt
    assert "not as a scene outline to reconstruct beat by beat" in normal_scene.generation_prompt
    assert "avoid modern analytical or process jargon" in fog_scene.generation_prompt


def test_scene_generator_does_not_award_unverified_lore_or_voice_scores():
    from book_graph_analyzer.generate.generator import SceneGenerator
    from book_graph_analyzer.generate.models import Scene, SceneScores

    generator = SceneGenerator()
    generator.judge = MagicMock()
    generator.judge.full_evaluation.return_value = (
        SceneScores(style_score=0.8, narrative_score=0.8),
        "",
        [],
    )
    scene = Scene(
        id="unverified",
        number=1,
        text="A road ran east.",
        pipeline_stages_run=["drafter"],
    )

    scores = generator._score_scene(scene, context="", lore_violations=[], voice_profiles={})

    assert scores.lore_score == 0.0
    assert scores.consistency_score == 0.0
    assert any("unverified" in note.lower() for note in scene.critique_notes)


def test_placeholder_world_bible_does_not_award_lore_score():
    from book_graph_analyzer.generate.context import AssembledContext
    from book_graph_analyzer.generate.generator import SceneGenerator
    from book_graph_analyzer.generate.models import Scene, SceneScores
    from book_graph_analyzer.generate.shadow.models import CharacterState
    from book_graph_analyzer.worldbible.models import WorldBible

    generator = SceneGenerator()
    generator.world_bible = WorldBible.from_markdown(
        "# Story Bible\n\n## World Rules\n- (add non-negotiable rules)\n"
    )
    generator._critique_scene = MagicMock(return_value=[])
    generator.judge = MagicMock()
    generator.judge.full_evaluation.return_value = (
        SceneScores(style_score=0.8, narrative_score=0.8),
        "",
        [],
    )
    scene = Scene(
        id="placeholder-bible",
        number=1,
        text="Beren walked east.",
        context_snapshot=AssembledContext(
            character_states=[CharacterState(name="Beren", story_id="story-1")],
            place_facts={"name": "Beleriand"},
        ),
    )

    scene, violations = generator.pipeline.run(
        scene,
        neo4j_context={},
        voice_profiles={},
    )
    scores = generator._score_scene(
        scene,
        context="",
        lore_violations=violations,
        voice_profiles={},
    )

    assert scene.pipeline_stages_run == ["drafter"]
    assert scores.lore_score == 0.0
    assert any("lore score is unverified" in note.lower() for note in scene.critique_notes)


def test_scene_generator_requires_an_attributed_profile_match_for_consistency_score():
    from book_graph_analyzer.generate.generator import SceneGenerator
    from book_graph_analyzer.generate.models import Scene, SceneScores
    from book_graph_analyzer.voice.profile import CharacterVoiceProfile

    generator = SceneGenerator()
    generator.judge = MagicMock()
    generator.judge.full_evaluation.return_value = (
        SceneScores(style_score=0.8, narrative_score=0.8),
        "",
        [],
    )
    bilbo_profile = CharacterVoiceProfile(
        character_name="Bilbo",
        contraction_ratio=0.0,
        formality_score=0.8,
        avg_utterance_length=5.0,
    )
    scene = Scene(
        id="no-profile-match",
        number=1,
        text='"The road runs east," said Beren.',
        characters=["Beren"],
        pipeline_stages_run=["drafter"],
    )

    scores = generator._score_scene(
        scene,
        context="",
        lore_violations=[],
        voice_profiles={"Bilbo": bilbo_profile},
    )

    assert scores.consistency_score == 0.0
    assert any("no attributed dialogue matched" in note.lower() for note in scene.critique_notes)


def test_scene_generator_scores_consistency_when_an_attributed_profile_matches():
    from book_graph_analyzer.generate.generator import SceneGenerator
    from book_graph_analyzer.generate.models import Scene, SceneScores
    from book_graph_analyzer.voice.profile import CharacterVoiceProfile

    generator = SceneGenerator()
    generator.judge = MagicMock()
    generator.judge.full_evaluation.return_value = (
        SceneScores(style_score=0.8, narrative_score=0.8),
        "",
        [],
    )
    beren_profile = CharacterVoiceProfile(
        character_name="Beren",
        contraction_ratio=0.0,
        formality_score=0.8,
        avg_utterance_length=4.0,
    )
    scene = Scene(
        id="profile-match",
        number=1,
        text='"The road runs east," said Beren.',
        characters=["Beren"],
        pipeline_stages_run=["drafter"],
    )

    scores = generator._score_scene(
        scene,
        context="",
        lore_violations=[],
        voice_profiles={"Beren": beren_profile},
    )

    assert scores.consistency_score > 0.0
    assert not any(
        "no attributed dialogue matched" in note.lower()
        for note in scene.critique_notes
    )


def test_scene_generator_normalizes_event_era_storage_in_temporal_query():
    from book_graph_analyzer.generate.generator import SceneGenerator

    generator = SceneGenerator()
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    generator.driver = driver

    empty_single = MagicMock()
    empty_single.single.return_value = None
    empty_rows = MagicMock()
    empty_rows.__iter__.return_value = iter([])
    event_queries: list[str] = []

    def run_side_effect(query, **_kwargs):
        if "MATCH (e:Event)" in query:
            event_queries.append(query)
            return empty_rows
        return empty_single

    session.run.side_effect = run_side_effect

    generator.get_context_from_neo4j(
        ["Aragorn"],
        "The Angle",
        story_era="Third Age",
        story_year=3017,
    )

    assert len(event_queries) == 1
    assert "replace(toLower(coalesce(e.era, '')), '_', ' ')" in event_queries[0]
    assert "replace(toLower($story_era), '_', ' ')" in event_queries[0]


def test_scene_generator_uses_canonical_name_fallback_from_graph():
    from book_graph_analyzer.generate.generator import SceneGenerator

    generator = SceneGenerator()

    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    generator.driver = driver

    char_result = MagicMock()
    char_result.single.return_value = {
        "name": "Gandalf",
        "type": "wizard",
        "desc": "A grey wanderer.",
        "relations": [{"rel": "KNOWS", "target": "Frodo Baggins"}],
    }
    place_result = MagicMock()
    place_result.single.return_value = {
        "name": "Rivendell",
        "desc": "A hidden valley.",
        "region": "Eriador",
    }
    empty_events = MagicMock()
    empty_events.__iter__.return_value = iter([])

    def run_side_effect(query, **_kwargs):
        if "MATCH (c:Character)" in query:
            return char_result
        if "MATCH (p:Place)" in query:
            return place_result
        return empty_events

    session.run.side_effect = run_side_effect

    context = generator.get_context_from_neo4j(["Gandalf"], "Rivendell")

    assert context["characters"][0]["name"] == "Gandalf"
    assert context["characters"][0]["relations"][0]["target"] == "Frodo Baggins"
    assert context["place"]["name"] == "Rivendell"
