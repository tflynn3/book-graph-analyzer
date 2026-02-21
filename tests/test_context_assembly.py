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
