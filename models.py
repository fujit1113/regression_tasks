"""
このファイルには、機械学習モデルを学習および評価するための抽象クラス _Model と、
それを継承した子クラス LgbModel, CatBoostModel が含まれています。

_Model:
    共通の学習および評価プロセスを定義する抽象クラス。

LgbModel:
    LightGBMを用いて回帰タスクを実行するクラス。

CatBoostModel:
    CatBoostを用いて回帰タスクを実行するクラス。

各子クラスは、ハイパーパラメータ探索とモデル評価のためのメソッドを持っています。
このファイルをインポートして使用することで、異なる機械学習アルゴリズムを簡単に試すことができます。
"""
import pickle
from abc import ABC, abstractmethod

from catboost import CatBoostRegressor
import numpy as np
import numba
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import optuna.integration.lightgbm as lgb
import shap
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb_ori

_RANDOM_SEED: int = 3655
_TEST_SPLIT: float = 0.2
_N_SPLIT: int = 5
_IS_SHUFFLED: bool = True
_EARLY_STOPPING_ROUNDS: int = 50
_NUM_BOOST_ROUND: int = 1000


class _Model(ABC):
    """
    _Modelクラスは、以降の回帰タスクで有効なクラスの親クラスです。

    Attributes:
        _random_seed (int): 乱数生成器のシード値。
        X_test (pd.DataFrame): テストデータの説明変数。
        y_test (pd.DataFrame): テストデータの目的変数。
        X_learn (pd.DataFrame): 学習データの説明変数。
        y_learn (pd.DataFrame): 学習データの目的変数。
        model (子クラスによる):
        _r2_eval (float): テストデータに対する決定係数（R^2）。
    """

    _random_seed: int
    X_test: pd.DataFrame = None
    y_test: pd.DataFrame = None
    X_learn: pd.DataFrame = None
    y_learn: pd.DataFrame = None
    _model = None
    _r2_eval: float = None
    kf: KFold = None

    def __init__(self, random_seed: int = _RANDOM_SEED):
        """
        コンストラクタ。シード値を設定します。

        Args:
            random_seed (int): 乱数生成器のシード値。
        """
        self._random_seed = random_seed

    def __lt__(self, other) -> bool:
        """
        自身と別のモデルのテストデータに対する予測精度を比較します。

        Args:
            other (_Model): 比較対象のインスタンス。

        Returns:
            bool: True 自身の予測精度が高い。 False 比較対象の予測精度が高い。
        """
        if isinstance(other, _Model):
            if (self.r2_eval is not None) and (other.r2_eval is not None):
                return self.__r2_eval > other.__r2_eval
            else:
                raise ValueError("モデルが未学習です。")
        else:
            raise TypeError(f"比較できない型です: {type(other)}")

    def __gt__(self, other) -> bool:
        """
        自身と別のモデルのテストデータに対する予測精度を比較します。

        Args:
            other (_Model): 比較対象のインスタンス。

        Returns:
            bool: True 自身の予測精度が低い。 False 比較対象の予測精度が低い。
        """
        if isinstance(other, _Model):
            if (self.__r2_eval is not None) and (other.__r2_eval is not None):
                return self.__r2_eval < other.__r2_eval
            else:
                raise ValueError("モデルが未学習です。")
        else:
            raise TypeError(f"比較できない型です: {type(other)}")

    def __call__(self, X: pd.DataFrame) -> np.array:
        """
        このクラスのインスタンスが関数のように呼ばれたとき、引数として与えられた説明変数に対する予測結果をかえします。

        Args:
            X (pd.DataFrame): 予測したい説明変数。

        Returns:
            Any: 予測結果。
        """
        return self.predict(X)

    def split_data(
        self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = _TEST_SPLIT
    ) -> None:
        """
        データをテストデータと学習データに分割します。

        Args:
            X (pd.DataFrame): 説明変数。
            y (pd.DataFrame): 目的変数。
            test_size (float): テストデータに割りあてる比率。
        """
        self.X_learn, self.X_test, self.y_learn, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._random_seed
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = _TEST_SPLIT,
        n_splits: int = _N_SPLIT,
        is_shuffled: bool = _IS_SHUFFLED,
        is_pickle_required: bool = False,
        **kwargs,
    ) -> None:
        """
        モデルが学習します。
        具体的な学習アルゴリズムは、子クラスの _train メソッドで実装します。

        Args:
            X (pd.DataFrame): 説明変数。
            y (pd.DataFrame): 目的変数。
            test_size (float): テストデータに割りあてる比率。
            n_splits: 交差検証の分割数。
            is_shuffled: 交差検証でデータ分割するときに、シャッフルするか否か。
            is_pickle_required (bool): pickle形式のファイルを作成するか否か。
            kwargs: 子クラスの _train メソッドに渡す追加のキーワード引数。
        """
        # テストデータと学習データに分割する
        self.split_data(X, y, test_size=test_size)

        # 交差検証の準備
        self.kf = KFold(
            n_splits=n_splits, shuffle=is_shuffled, random_state=self._random_seed
        )

        # 子クラスごとに定義する
        self._train(**kwargs)

        # pkl形式で保存
        if is_pickle_required:
            with open("model/model.pkl", "wb") as f:
                pickle.dump(self._model, f)

    @abstractmethod
    def _train(self) -> None:
        """
        モデルが学習します。
        継承先のクラスで実装してください。
        """
        pass

    def _evaluate_models(self):
        """
        以下の2つのモデルを評価します。

        1. 全学習データで再学習したモデル
        2. ハイパーパラメータ探索と同じデータ数で学習し、平均したモデル

        これら2つの方法で評価した2モデルのうち、テストデータに対して高精度なモデルを
        アトリビュートに登録します。
        """
        # 1. 全学習データで再学習したモデル
        model_all = self._train_with_best_params(self.X_learn, self.y_learn)
        r2_all = r2_score(self.y_test, model_all.predict(self.X_test))

        # 2. ハイパーパラメータ探索と同じデータ数で学習し、平均したモデル
        y_pred_mean = np.zeros(len(self.y_test), dtype=np.float64)
        models = []
        for train_idx, valid_idx in self.kf.split(self.X_learn):
            X_train, X_valid = (
                self.X_learn.iloc[train_idx],
                self.X_learn.iloc[valid_idx],
            )
            y_train, y_valid = (
                self.y_learn.iloc[train_idx],
                self.y_learn.iloc[valid_idx],
            )

            model = self._train_with_best_params(X_train, y_train, X_valid, y_valid)
            y_pred_mean += model.predict(self.X_test)
            models.append(model)
        y_pred_mean /= self.kf.get_n_splits()
        r2_mean = r2_score(self.y_test, y_pred_mean)

        # テストデータに対して高精度なモデルをアトリビュートに登録
        if r2_all >= r2_mean:
            self._model, self._r2_eval = [model_all], r2_all
        else:
            self._model, self._r2_eval = models, r2_mean

    @abstractmethod
    def _train_with_best_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ):
        """
        最適なハイパーパラメータをつかって、モデルが再学習します。
        継承先のクラスで実装してください。

        Args:
            X_train (pd.DataFrame): 学習データの説明変数。
            y_train (pd.DataFrame): 学習データの目的変数。
            X_valid (pd.DataFrame, optional): 検証データの説明変数。
            y_valid (pd.DataFrame, optional): 検証データの目的変数。

        Returns:
            Union[Booster, CatBoostRegressor]: 学習済みのモデル。
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        モデルが予測します。

        Args:
            X (pd.DataFrame): 説明変数。
        """
        if self.model is None:
            raise ValueError("モデルが未学習です。")

        y_pred = np.zeros(len(X), dtype=np.float64)
        for model in self.model:
            y_pred += model.predict(X)

        return y_pred / len(self.model)

    def r2_plot(
        self,
        x_label: str = "x",
        learn_color: str = "lightgrey",
        test_color: str = "lightcoral",
        line_color: str = "grey",
    ) -> None:
        """
        学習データとテストデータの予測結果を目的変数と比較し、予測精度、過学習していないかを目視します。

        Args:
            x_label (str): x軸のラベル。
            learn_color (str): 学習データのプロットの色。
            test_color (str):　テストデータのプロットの色。
            line_color (str): 目安線 (相関係数 1.0 の線) の色。
        """
        # 学習データとその予測の相関を求める
        y_pred_learn = self.predict(self.X_learn)
        r2_learn = r2_score(self.y_learn, y_pred_learn)
        val_max_learn = max(max(self.y_learn), max(y_pred_learn))
        val_min_learn = min(min(self.y_learn), min(y_pred_learn))

        # テストデータとその予測の相関を求める
        y_pred_test = self.predict(self.X_test)
        val_max_test = max(max(self.y_test), max(y_pred_test))
        val_min_test = min(min(self.y_test), min(y_pred_test))

        # 相関係数が 1.0 のときの直線
        ideal_line = [
            val
            for val in np.linspace(
                min(val_min_test, val_min_learn), max(val_max_test, val_max_learn), 100
            )
        ]

        # 描画する
        _, ax = plt.subplots()
        ax.set_title("Relation")
        ax.set_xlabel(x_label)
        ax.set_ylabel("pred")
        ax.plot(ideal_line, ideal_line, color=line_color, linestyle="dashed")
        ax.scatter(
            self.y_learn,
            y_pred_learn,
            color=learn_color,
            label=r"learn $R^2$: " + f"{r2_learn:.2f}",
        )
        ax.scatter(
            self.y_test,
            y_pred_test,
            color=test_color,
            label=r"test $R^2$: " + f"{self._r2_eval:.2f}",
        )
        ax.legend()
        ax.set_aspect("equal")

    def calculate_shap_values(self, X_sample: pd.DataFrame):
        """
        学習後に、モデルを使って予測する。

        Args:
            X_sample (pd.DataFrame): 説明変数。
        """
        explainer = shap.Explainer(self.model)
        return explainer(X_sample)


class LgbModel(_Model):
    """
    このクラスは、lightGBM で回帰タスクを実施します。

    Attributes:
        best_params (dict): モデルのハイパーパラメータ。
        params (dict): ハイパーパラメータ探索に必要なパラメータ。
        early_stopping_rounds (int): 予測性能が改善されないときの学習停止回数。
        num_boost_round (int): ブースティングの繰り返し回数。
    """

    best_params: dict = None
    params: dict = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "rmse",
        "n_jobs": -1,
        "verbose": 0,
    }
    early_stopping_rounds: int = None
    num_boost_round: int = None

    def _train(
        self,
        early_stopping_rounds: int = _EARLY_STOPPING_ROUNDS,
        num_boost_round: int = _NUM_BOOST_ROUND,
    ) -> None:
        """
        lightGBM で学習します。
        最適なハイパーパラメータを LightGBMTunerCV で取得します。

        args:
            early_stopping_rounds (int): 予測性能が改善されないときの学習停止回数。
            num_boost_round (int): ブースティングの繰り返し回数。
        """
        # 過学習防止のハイパーパラメータ
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round

        # LightGBMTunerCV の再現性のため
        sampler = optuna.samplers.TPESampler(seed=self._random_seed)
        study = optuna.create_study(sampler=sampler, direction="minimize")

        # 学習
        lgb_learn = lgb.Dataset(self.X_learn, self.y_learn)
        tuner = lgb.LightGBMTunerCV(
            self.params,
            lgb_learn,
            early_stopping_rounds=self.early_stopping_rounds,
            num_boost_round=self.num_boost_round,
            folds=self.kf,
            study=study,
        )
        tuner.run()
        self.best_params = tuner.best_params

        # 最適なハイパーパラメータでモデル評価
        self._evaluate_models()

    def _train_with_best_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ) -> lgb.Booster:
        """
        最適なハイパーパラメータでlightGBMモデルが学習します。

        Args:
            X_train (pd.DataFrame): 訓練データの説明変数。
            y_train (pd.Series): 訓練データの目的変数。
            X_valid (pd.DataFrame, optional): 検証データの説明変数。
            y_valid (pd.Series, optional): 検証データの目的変数。

        Returns:
            lgb.Booster: 学習ずみの lightGBM モデル
        """
        lgb_train = lgb.Dataset(X_train, label=y_train)
        if (X_valid is not None) and (y_valid is not None):
            lgb_valid = lgb.Dataset(X_valid, label=y_valid)
            valid_sets = [lgb_valid]
            model = lgb.train(
                self.best_params,
                lgb_train,
                valid_sets=valid_sets,
                early_stopping_rounds=self.early_stopping_rounds,
                num_boost_round=self.num_boost_round,
                verbose_eval=False,
            )
        else:
            model = lgb_ori.train(
                self.best_params,
                lgb_train,
                num_boost_round=self.num_boost_round,
                verbose_eval=False,
            )

        return model


class CatBoostModel(_Model):
    """
    このクラスは、CatBoost で回帰タスクを実施します。

    Attributes:
        best_params (dict): モデルのハイパーパラメータ。
        params (dict): ハイパーパラメータ探索に必要なパラメータ。
        early_stopping_rounds (int): 予測性能が改善されないときの学習停止回数。
    """

    best_params: dict = None
    params: dict = {
        "random_seed": None,
        "logging_level": "Silent",
        "eval_metric": "RMSE",
        "od_type": "Iter",
        "od_wait": None,
    }
    params_tuned: dict = {
        "iterations": optuna.distributions.IntDistribution(100, 1000),
        "depth": optuna.distributions.IntDistribution(1, 10),
        "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1),
        "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-8, 100),
        "border_count": optuna.distributions.IntDistribution(1, 255),
    }
    early_stopping_rounds: int = None

    def __init__(
        self,
        random_seed: int = _RANDOM_SEED,
    ):
        """
        コンストラクタ。シード値を設定します。

        Args:
            random_seed (int): 乱数生成器のシード値。
        """
        super().__init__(random_seed)
        self.params["random_seed"] = random_seed

    def _train(
        self,
        early_stopping_rounds: int = _EARLY_STOPPING_ROUNDS,
    ) -> None:
        """
        CatBoost で学習します。
        最適なハイパーパラメータを OptunaSearchCV で取得します。

        args:
            early_stopping_rounds (int): 予測性能が改善されないときの学習停止回数。
        """
        # 過学習防止のハイパーパラメータ
        self.params["od_wait"] = early_stopping_rounds

        # モデル生成の再現性のため
        sampler = optuna.samplers.TPESampler(seed=self._random_seed)
        study = optuna.create_study(sampler=sampler, direction="maximize")

        # 学習
        model = CatBoostRegressor(**self.params)
        optuna_search = OptunaSearchCV(
            model,
            self.params_tuned,
            cv=self.kf,
            scoring="neg_root_mean_squared_error",
            random_state=self._random_seed,
            study=study,
            verbose=0,
        )
        optuna_search.fit(self.X_learn, self.y_learn)
        self.best_params = optuna_search.best_params_

        # 最適なハイパーパラメータでモデル評価
        self._evaluate_models()

    def _train_with_best_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ) -> CatBoostRegressor:
        """
        最適なハイパーパラメータで CatBoost モデルが学習します。

        Args:
            X_train (pd.DataFrame): 訓練データの説明変数。
            y_train (pd.Series): 訓練データの目的変数。
            X_valid (pd.DataFrame, optional): 検証データの説明変数。
            y_valid (pd.Series, optional): 検証データの目的変数。

        Returns:
            CatBoostRegressor: 学習済みの CatBoost モデル
        """
        model = CatBoostRegressor(
            **self.params,
            **self.best_params,
        )

        if (X_valid is not None) and (y_valid is not None):
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
            )
        else:
            model.fit(X_train, y_train)

        return model


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import r2_score

    # データセットのロード
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")

    # LgbModelのデモ
    lgb_model = LgbModel()
    lgb_model.train(X, y)

    # CatBoostModelのデモ
    cat_model = CatBoostModel()
    cat_model.train(X, y)

    # 結果の表示
    print("LgbModel R^2: ", lgb_model._r2_eval)
    print("CatBoostModel R^2: ", cat_model._r2_eval)
