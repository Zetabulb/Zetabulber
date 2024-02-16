#ifndef __TEST_SCENARIOS__
#define __TEST_SCENARIOS__

#include <vector>
#include "params.cuh"

class TestScenario {
public:
	float horizontalRotation[3] = {};
	float verticalRotation = 0.0f;
	float zetaPosition[2] = { };
	float position[3] = {};
	float power = 0.0f;
	bool re_movement_enabled;
	bool im_movement_enabled;
	bool zeta_enabled;
	bool power_enabled;
};

class TestScenarios {
private:
	std::vector<TestScenario> _test_scenarios;
public:
	void addScenario(TestScenario scenario) {
		_test_scenarios.push_back(scenario);
	}

	bool scenarioExists(int index) {
		return index < _test_scenarios.size();
	}

	void applyScenario(SceneParams &params, int index) {
		TestScenario scenario = _test_scenarios.at(index);

		params.camera.setPosition(scenario.position);
		params.camera.setRotation(scenario.horizontalRotation, scenario.verticalRotation);
		params.set_zeta_position(scenario.zetaPosition);
		params.toggle_im_movement(scenario.im_movement_enabled);
		params.toggle_re_movement(scenario.re_movement_enabled);
		params.toggle_zeta(scenario.zeta_enabled);
		params.toggle_power(scenario.power_enabled);
		params.set_power(scenario.power);
	}

	void initDefault() {
		TestScenario scenario1;
		scenario1.horizontalRotation[0] = 0.54222f;
		scenario1.horizontalRotation[1] = 0.0f;
		scenario1.horizontalRotation[2] = -0.965913f;
		scenario1.verticalRotation = 0.011f;
		scenario1.zetaPosition[0] = 1.5f;
		scenario1.zetaPosition[1] = 5.0f;
		scenario1.position[0] = -0.809405f;
		scenario1.position[1] = 0.3f;
		scenario1.position[2] = 1.81017f;
		scenario1.power = 8.0f;
		scenario1.re_movement_enabled = false;
		scenario1.im_movement_enabled = false;
		scenario1.zeta_enabled = false;
		scenario1.power_enabled = false;
		addScenario(scenario1);

		TestScenario scenario2;
		scenario2.horizontalRotation[0] = 0.54222f;
		scenario2.horizontalRotation[1] = 0.0f;
		scenario2.horizontalRotation[2] = -0.965913f;
		scenario2.verticalRotation = 0.011f;
		scenario2.zetaPosition[0] = 1.5f;
		scenario2.zetaPosition[1] = 5.0f;
		scenario2.position[0] = -0.809405f;
		scenario2.position[1] = 0.3f;
		scenario2.position[2] = 1.81017f;
		scenario2.power = 8.0f;
		scenario2.re_movement_enabled = false;
		scenario2.im_movement_enabled = false;
		scenario2.zeta_enabled = false;
		scenario2.power_enabled = true;
		addScenario(scenario2);

		TestScenario scenario3;
		scenario3.horizontalRotation[0] = 0.54222f;
		scenario3.horizontalRotation[1] = 0.0f;
		scenario3.horizontalRotation[2] = -0.965913f;
		scenario3.verticalRotation = 0.011f;
		scenario3.zetaPosition[0] = 1.5f;
		scenario3.zetaPosition[1] = 5.0f;
		scenario3.position[0] = -0.809405f;
		scenario3.position[1] = 0.3f;
		scenario3.position[2] = 1.81017f;
		scenario3.power = 8.0f;
		scenario3.re_movement_enabled = false;
		scenario3.im_movement_enabled = false;
		scenario3.zeta_enabled = true;
		scenario3.power_enabled = false;
		addScenario(scenario3);

		TestScenario scenario4;
		scenario4.horizontalRotation[0] = 0.54222f;
		scenario4.horizontalRotation[1] = 0.0f;
		scenario4.horizontalRotation[2] = -0.965913f;
		scenario4.verticalRotation = 0.011f;
		scenario4.zetaPosition[0] = 1.5f;
		scenario4.zetaPosition[1] = 5.0f;
		scenario4.position[0] = -0.809405f;
		scenario4.position[1] = 0.3f;
		scenario4.position[2] = 1.81017f;
		scenario4.power = 8.0f;
		scenario4.re_movement_enabled = false;
		scenario4.im_movement_enabled = true;
		scenario4.zeta_enabled = true;
		scenario4.power_enabled = false;
		addScenario(scenario4);
	}
};

#endif